import asyncio
import json
from pathlib import Path
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.responses import FunctionToolParam, Response, ResponseInputItemParam

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

SHELL_TOOL: FunctionToolParam = {
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
}


# added this on AI suggestion to improve token efficiency. keeps the first and last 10k chars of a
# long shell output
def _truncate(text: str) -> str:
    """Keep both ends of long output, where errors and summaries usually appear."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    omitted = len(text) - MAX_OUTPUT_CHARS
    return f"{text[:half]}\n... [{omitted} characters omitted] ...\n{text[-half:]}"


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


async def _call_tool(name: str, arguments: str, cwd: Path) -> str:
    if name != "shell":
        return f"Unknown tool {name!r}; the only tool is 'shell'."
    try:
        command = json.loads(arguments)["command"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return "Invalid arguments; expected a JSON object with a string 'command'."
    if not isinstance(command, str):
        return "Invalid arguments; 'command' must be a string."
    return await run_shell(command, cwd)


class WebpAgent:
    """Loop between the model and a shell tool until the model stops calling tools."""

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self._client = client

    async def run(self, instruction: str, context: RunContext) -> str:
        settings = UNCOpenAISettings.from_env(model=context.model)
        # if client exists start loop
        if self._client is not None:
            return await self._loop(self._client, settings.model, instruction, context)
        # otherwise make a client w proper cleanup
        async with settings.create_async_client() as client:
            patient = client.with_options(timeout=REQUEST_TIMEOUT_SECONDS, max_retries=2)
            return await self._loop(patient, settings.model, instruction, context)

    async def _loop(
        self, client: AsyncOpenAI, model: str, instruction: str, context: RunContext
    ) -> str:
        transcript: list[ResponseInputItemParam] = [{"role": "user", "content": instruction}]
        for _ in range(MAX_STEPS):
            # call model first
            response: Response = await client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT.format(cwd=context.cwd),
                input=transcript,
                tools=[SHELL_TOOL],
                include=["reasoning.encrypted_content"],
                store=False,
                background=False,
            )
            # replays every output item to maintain context for the model between turns
            for item in response.output:
                transcript.append(cast(ResponseInputItemParam, item.model_dump(exclude_none=True)))
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:  # task finished, terminate
                return response.output_text
            # otherwise execute calls
            for call in calls:
                output = await _call_tool(call.name, call.arguments, context.cwd)
                result: dict[str, Any] = {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": output,
                }
                transcript.append(cast(ResponseInputItemParam, result))
        return f"Stopped after reaching the {MAX_STEPS}-step limit."
