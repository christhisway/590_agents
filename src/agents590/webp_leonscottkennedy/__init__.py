import asyncio
import json
import os
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from openai import APIError, AsyncOpenAI
from openai.types.responses import (
    FunctionToolParam,
    Response,
    ResponseFunctionToolCall,
    ResponseInputItemParam,
    ResponseOutputItem,
    ResponseReasoningItem,
)

from agents590.atif import ATIFRecorder, JSONValue
from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings

MAX_STEPS = 50
MAX_STEPS_ENV = "AGENTS590_MAX_STEPS"
COMMAND_TIMEOUT_SECONDS = 120.0
MAX_OUTPUT_CHARS = 20_000
REQUEST_TIMEOUT_SECONDS = 300.0


def _configured_max_steps() -> int:
    """Read the step limit, requiring an explicitly positive integer when set."""
    value = os.environ.get(MAX_STEPS_ENV)
    if value is None:
        return MAX_STEPS
    if not value.isdecimal() or int(value) <= 0:
        raise ValueError(f"{MAX_STEPS_ENV} must be a positive integer; got {value!r}.")
    return int(value)


# not sure if this is even necessary but it couldn't hurt
SYSTEM_PROMPT = """\
You are an autonomous coding agent working in a Linux or macOS shell. Nobody will \
answer questions, so complete the task end to end without asking for confirmation.

Use the `shell` tool to inspect files, edit them, and run code or tests. Each call \
runs in a fresh `bash -c` with the working directory {cwd}; state such as `cd` does \
not persist between calls. Prefer small, precise edits (for example a Python script, \
`sed`, or a heredoc that rewrites a whole file you have just read). Change only what \
the task requires and preserve unrelated content exactly.

Before finishing, verify your change by running it or its tests where possible. \
When the task is complete, reply with a short summary and no tool call."""

# Repository conventions the agent should follow without being reminded in each instruction.
PROJECT_INSTRUCTIONS_FILE = "AGENTS.md"

# Present only in a checkout of this agent's own repository; the shared `make test` skips
# the agent's tests, so the prompt points at this script whenever it exists under cwd.
SELF_TEST_SCRIPT = Path("src/agents590/webp_leonscottkennedy/test.sh")
SELF_TEST_NOTE = """

This repository contains your own source code. To run the tests, use `{script}` in place \
of `make test`; it also runs your own tests, which `make test` skips."""

# Sent when a response is cut off at the output-token limit without calling a tool.
CONTINUE_PROMPT = "Your previous response was cut off at the output limit. Continue the task."


# added this on AI suggestion to improve token efficiency. keeps the first and last 10k chars of a
# long shell output
def _truncate(text: str) -> str:
    """Keep both ends of long output, where errors and summaries usually appear."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    omitted = len(text) - MAX_OUTPUT_CHARS
    return f"{text[:half]}\n... [{omitted} characters omitted] ...\n{text[-half:]}"


def _system_prompt(cwd: Path) -> str:
    """Append the self-test note and the working directory's AGENTS.md, when present."""
    prompt = SYSTEM_PROMPT.format(cwd=cwd)
    if (cwd / SELF_TEST_SCRIPT).is_file():
        prompt += SELF_TEST_NOTE.format(script=SELF_TEST_SCRIPT)
    path = cwd / PROJECT_INSTRUCTIONS_FILE
    if not path.is_file():
        return prompt
    instructions = _truncate(path.read_text(errors="replace").strip())
    return (
        f"{prompt}\n\nThe project's {PROJECT_INSTRUCTIONS_FILE} follows. Treat it as standing "
        f"instructions for every change you make.\n\n{instructions}"
    )


async def run_shell(command: str, cwd: Path, timeout: float = COMMAND_TIMEOUT_SECONDS) -> str:
    process = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        command,
        cwd=cwd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        # The shell can have spawned children (for example, pytest).  Put each
        # command in its own session so that a timeout cannot leave those
        # children running after the shell is killed.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
        return f"Command timed out after {timeout:g} seconds."
    return _truncate(
        f"exit_code: {process.returncode}\n"
        f"stdout:\n{stdout.decode(errors='replace')}\n"
        f"stderr:\n{stderr.decode(errors='replace')}"
    )


@dataclass(frozen=True)
class Tool:
    """A function tool's schema and the coroutine that runs it on parsed arguments."""

    schema: FunctionToolParam
    run: Callable[[dict[str, Any], Path], Awaitable[str]]


async def _shell(arguments: dict[str, Any], cwd: Path) -> str:
    command = arguments.get("command")
    if not isinstance(command, str):
        return "Invalid arguments; 'command' must be a string."
    return await run_shell(command, cwd)


SHELL = Tool(
    schema={
        "type": "function",
        "name": "shell",
        "description": "Run a bash command and return its exit code, stdout, and stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute."},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    run=_shell,
)

TOOLS: dict[str, Tool] = {tool.schema["name"]: tool for tool in [SHELL]}


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed: object = json.loads(text)
    except json.JSONDecodeError:
        return None
    return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None


async def _call_tool(name: str, arguments: str, cwd: Path) -> str:
    tool = TOOLS.get(name)
    if tool is None:
        return f"Unknown tool {name!r}; available tools: {', '.join(TOOLS)}."
    parsed = _json_object(arguments)
    if parsed is None:
        return "Invalid arguments; expected a JSON object."
    return await tool.run(parsed, cwd)


async def _request(
    client: AsyncOpenAI, model: str, system_prompt: str, transcript: list[ResponseInputItemParam]
) -> Response:
    """Ask for the next turn statelessly, carrying encrypted reasoning in the transcript."""
    return await client.responses.create(
        model=model,
        instructions=system_prompt,
        input=transcript,
        tools=[tool.schema for tool in TOOLS.values()],
        include=["reasoning.encrypted_content"],
        store=False,
        background=False,
    )


async def _run_calls(calls: list[ResponseFunctionToolCall], cwd: Path) -> list[tuple[str, str]]:
    """Run each tool call in order and pair its output with the call's id."""
    return [(call.call_id, await _call_tool(call.name, call.arguments, cwd)) for call in calls]


def _as_input(item: ResponseOutputItem) -> ResponseInputItemParam:
    """Replay a model output item verbatim so the next request keeps the full context."""
    return cast(ResponseInputItemParam, item.model_dump(exclude_none=True))


def _call_output(call_id: str, output: str) -> ResponseInputItemParam:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _arguments_object(arguments: str) -> dict[str, Any]:
    """ATIF requires an object, so keep malformed model arguments under 'raw'."""
    parsed = _json_object(arguments)
    return parsed if parsed is not None else {"raw": arguments}


def _record_response(
    recorder: ATIFRecorder, response: Response, outputs: list[tuple[str, str]]
) -> None:
    """Record one model response with the (call_id, output) of each tool it called."""
    fields: dict[str, JSONValue] = {"model_name": response.model}
    reasoning = [
        part.text
        for item in response.output
        if isinstance(item, ResponseReasoningItem)
        for part in item.summary
    ]
    if reasoning:
        fields["reasoning_content"] = "\n\n".join(reasoning)
    calls = [item for item in response.output if isinstance(item, ResponseFunctionToolCall)]
    if calls:
        fields["tool_calls"] = [
            {
                "tool_call_id": call.call_id,
                "function_name": call.name,
                "arguments": cast(JSONValue, _arguments_object(call.arguments)),
            }
            for call in calls
        ]
        fields["observation"] = {
            "results": [
                {"source_call_id": call_id, "content": output} for call_id, output in outputs
            ]
        }
    if response.usage is not None:
        fields["metrics"] = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "cached_tokens": response.usage.input_tokens_details.cached_tokens,
        }
    recorder.add("agent", response.output_text, **fields)


def _stop_reason(response: Response) -> str | None:
    """Explain a response that must end the run; a max_output_tokens cut-off can continue."""
    if response.status == "failed":
        detail = response.error.message if response.error is not None else "no details"
        return f"the model response failed ({detail})"
    if response.status == "incomplete":
        details = response.incomplete_details
        reason = details.reason if details is not None else None
        if reason != "max_output_tokens":
            return f"the model response was incomplete ({reason or 'unknown reason'})"
    return None


def _stop(recorder: ATIFRecorder, reason: str) -> str:
    """Record why the run ended early and return that as the final output."""
    message = f"Stopped early: {reason}."
    recorder.add("system", message)
    return message


class LeonScottKennedy:
    """Loop between the model and a shell tool until the model stops calling tools."""

    def __init__(
        self, client: AsyncOpenAI | None = None, trajectory_path: Path | None = None
    ) -> None:
        self._client = client
        self._trajectory_path = trajectory_path

    async def run(self, instruction: str, context: RunContext) -> str:
        max_steps = _configured_max_steps()
        settings = UNCOpenAISettings.from_env(model=context.model)
        # if client exists start loop
        if self._client is not None:
            return await self._loop(self._client, settings.model, instruction, context, max_steps)
        # otherwise make a client w proper cleanup
        async with settings.create_async_client() as client:
            patient = client.with_options(timeout=REQUEST_TIMEOUT_SECONDS, max_retries=2)
            return await self._loop(patient, settings.model, instruction, context, max_steps)

    def _recorder(self, model: str) -> ATIFRecorder:
        if self._trajectory_path is not None:
            return ATIFRecorder(self._trajectory_path, agent="webp_leonscottkennedy", model=model)
        return ATIFRecorder.from_environment(agent="webp_leonscottkennedy", model=model)

    async def _loop(
        self,
        client: AsyncOpenAI,
        model: str,
        instruction: str,
        context: RunContext,
        max_steps: int,
    ) -> str:
        system_prompt = _system_prompt(context.cwd)
        recorder = self._recorder(model)
        recorder.add("system", system_prompt)
        recorder.add("user", instruction)
        transcript: list[ResponseInputItemParam] = [{"role": "user", "content": instruction}]
        for _ in range(max_steps):
            try:
                response = await _request(client, model, system_prompt, transcript)
            except APIError as error:
                # Covers context overflow, exhausted retries, and connection failures.
                failure = f"{type(error).__name__}: {error}"
                return _stop(recorder, f"the model request failed ({failure})")
            # Don't run tool calls from a failed or filtered response.
            if (reason := _stop_reason(response)) is not None:
                _record_response(recorder, response, [])
                return _stop(recorder, reason)
            calls = [item for item in response.output if isinstance(item, ResponseFunctionToolCall)]
            outputs = await _run_calls(calls, context.cwd)
            transcript += [_as_input(item) for item in response.output]
            transcript += [_call_output(call_id, output) for call_id, output in outputs]
            _record_response(recorder, response, outputs)
            if not calls and response.status == "incomplete":
                # Cut off at max_output_tokens: the text so far isn't a final answer.
                transcript.append({"role": "user", "content": CONTINUE_PROMPT})
                recorder.add("user", CONTINUE_PROMPT)
            elif not calls:
                return response.output_text
        return f"Stopped after reaching the {max_steps}-step limit."
