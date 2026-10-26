from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.responses import (
    Response,
    ResponseInputItemParam,
    ResponseOutputItem,
    ToolParam,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput

from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings


class MyAgent:
    """Scaffold for a model/tool loop using the shared standalone interface."""

    client: AsyncOpenAI
    settings: UNCOpenAISettings

    def provider_settings(self, context: RunContext) -> UNCOpenAISettings:
        """Prefer the per-run model to the environment's configured deployment."""
        return UNCOpenAISettings.from_env(model=context.model)

    tools: list[ToolParam] = cast(
        list[ToolParam],
        [
            {
                "type": "function",
                "name": "run_shell_command",
                "description": "Run arbitrary shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "string",
                            "description": "Command to execute, ex. ls",
                        },
                    },
                    "required": ["cmd"],
                },
            },
        ],
    )

    async def run(self, instruction: str, context: RunContext) -> str:
        self.settings = self.provider_settings(context)
        self.client = self.settings.create_async_client()
        response = await self.call_model(instruction)
        return await self.handle_response([], response.output, context.cwd)

    async def handle_response(
        self,
        old_input: list[ResponseInputItemParam],
        output: Iterable[ResponseOutputItem],
        cwd: Path,
    ) -> str:
        """Handle a Responses result and continue until a message completes."""
        output = list(output)
        if not output:
            raise RuntimeError("The model returned an empty response.")

        append_to_input: list[ResponseInputItemParam] = []

        for output_item in output:
            match output_item.type:
                case "function_call":
                    try:
                        parsed_args = json.loads(output_item.arguments)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid arguments for tool {output_item.name!r}: "
                            f"{output_item.arguments!r}"
                        ) from error
                    if not isinstance(parsed_args, dict):
                        raise ValueError(
                            f"Arguments for tool {output_item.name!r} must be an object"
                        )
                    args = cast(dict[str, Any], parsed_args)
                    result = await self.call_function(output_item.name, args, cwd)
                    append_to_input.append(
                        FunctionCallOutput(
                            type="function_call_output",
                            call_id=output_item.call_id,
                            output=json.dumps(result),
                        )
                    )
                case "message":
                    text = "".join(
                        part.text for part in output_item.content if part.type == "output_text"
                    )
                    if output_item.status == "completed":
                        return text
                case "reasoning":
                    continue
                case _:
                    raise RuntimeError(f"Unhandled response item type: {output_item.type}")

        prior_output = [
            cast(ResponseInputItemParam, item.model_dump(exclude_none=True)) for item in output
        ]
        next_input = [*old_input, *prior_output, *append_to_input]
        response = await self.call_model(next_input)
        return await self.handle_response(next_input, response.output, cwd)

    async def call_model(self, prompt: str | list[ResponseInputItemParam]) -> Response:
        return await self.client.responses.create(
            model=self.settings.model,
            instructions="",
            tools=self.tools,
            input=prompt,
            store=False,
            background=False,
        )

    async def call_function(
        self,
        name: str,
        args: dict[str, Any],
        cwd: Path,
    ) -> str:
        if name == "run_shell_command":
            cmd = args.get("cmd")
            if not isinstance(cmd, str):
                raise ValueError("run_shell_command requires a string 'cmd' argument")
            return await self.run_shell_command(cmd, cwd)
        raise ValueError(f"Unknown function: {name}")

    async def run_shell_command(self, cmd: str, cwd: Path | None = None) -> str:
        def run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=False,
                cwd=str(cwd) if cwd is not None else None,
            )

        result = await asyncio.to_thread(run)

        if result.returncode == 0:
            return result.stdout
        return (
            f"Command exited with status {result.returncode}.\n"
            f"stdout:\n{result.stdout}"
            f"stderr:\n{result.stderr}"
        )
