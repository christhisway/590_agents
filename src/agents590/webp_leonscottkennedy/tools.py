"""Tool definitions and their local implementations."""

import asyncio
import json
import os
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from openai.types.responses import FunctionToolParam

from . import skills
from .prompts import truncate

COMMAND_TIMEOUT_SECONDS = 120.0


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
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
        return f"Command timed out after {timeout:g} seconds."
    return truncate(
        f"exit_code: {process.returncode}\n"
        f"stdout:\n{stdout.decode(errors='replace')}\n"
        f"stderr:\n{stderr.decode(errors='replace')}"
    )


@dataclass(frozen=True)
class Tool:
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
                "command": {"type": "string", "description": "The bash command to execute."}
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    run=_shell,
)


def _skill_tool(found: list[skills.Skill]) -> Tool:
    by_name = {skill.name: skill for skill in found}
    activated: set[str] = set()

    async def activate(arguments: dict[str, Any], cwd: Path) -> str:
        name = arguments.get("name")
        if not isinstance(name, str) or name not in by_name:
            return f"Unknown skill {name!r}; available skills: {', '.join(by_name)}."
        if name in activated:
            return f"Skill {name!r} is already active; its instructions are above."
        activated.add(name)
        return skills.activate(by_name[name])

    return Tool(
        schema={
            "type": "function",
            "name": "activate_skill",
            "description": "Load a skill's full instructions and list its bundled files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": list(by_name),
                        "description": "The name of the skill to activate.",
                    }
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        run=activate,
    )


def tools_for(found: list[skills.Skill]) -> dict[str, Tool]:
    available = [SHELL, _skill_tool(found)] if found else [SHELL]
    return {tool.schema["name"]: tool for tool in available}


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed: object = json.loads(text)
    except json.JSONDecodeError:
        return None
    return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None


async def call_tool(tools: dict[str, Tool], name: str, arguments: str, cwd: Path) -> str:
    tool = tools.get(name)
    if tool is None:
        return f"Unknown tool {name!r}; available tools: {', '.join(tools)}."
    parsed = _json_object(arguments)
    if parsed is None:
        return "Invalid arguments; expected a JSON object."
    return await tool.run(parsed, cwd)
