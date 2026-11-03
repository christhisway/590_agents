import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from openai import AsyncOpenAI
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
COMMAND_TIMEOUT_SECONDS = 120.0
MAX_OUTPUT_CHARS = 20_000
REQUEST_TIMEOUT_SECONDS = 300.0

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
    """Append the working directory's AGENTS.md, if any, to the base prompt."""
    prompt = SYSTEM_PROMPT.format(cwd=cwd)
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
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        process.kill()
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


class LeonScottKennedy:
    """Loop between the model and a shell tool until the model stops calling tools."""

    def __init__(
        self, client: AsyncOpenAI | None = None, trajectory_path: Path | None = None
    ) -> None:
        self._client = client
        self._trajectory_path = trajectory_path

    async def run(self, instruction: str, context: RunContext) -> str:
        settings = UNCOpenAISettings.from_env(model=context.model)
        # if client exists start loop
        if self._client is not None:
            return await self._loop(self._client, settings.model, instruction, context)
        # otherwise make a client w proper cleanup
        async with settings.create_async_client() as client:
            patient = client.with_options(timeout=REQUEST_TIMEOUT_SECONDS, max_retries=2)
            return await self._loop(patient, settings.model, instruction, context)

    def _recorder(self, model: str) -> ATIFRecorder:
        if self._trajectory_path is not None:
            return ATIFRecorder(self._trajectory_path, agent="webp_leonscottkennedy", model=model)
        return ATIFRecorder.from_environment(agent="webp_leonscottkennedy", model=model)

    async def _loop(
        self, client: AsyncOpenAI, model: str, instruction: str, context: RunContext
    ) -> str:
        system_prompt = _system_prompt(context.cwd)
        recorder = self._recorder(model)
        recorder.add("system", system_prompt)
        recorder.add("user", instruction)
        transcript: list[ResponseInputItemParam] = [{"role": "user", "content": instruction}]
        for _ in range(MAX_STEPS):
            response = await _request(client, model, system_prompt, transcript)
            calls = [item for item in response.output if isinstance(item, ResponseFunctionToolCall)]
            outputs = await _run_calls(calls, context.cwd)
            transcript += [_as_input(item) for item in response.output]
            transcript += [_call_output(call_id, output) for call_id, output in outputs]
            _record_response(recorder, response, outputs)
            if not calls:
                return response.output_text
        return f"Stopped after reaching the {MAX_STEPS}-step limit."
