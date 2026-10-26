from __future__ import annotations

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
    orig_prompt: str

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
        self.orig_prompt = instruction

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
        append_to_input: list[ResponseInputItemParam] = []

        for output_item in output:
            match output_item.type:
                case "function_call":
                    args: dict[str, Any] = json.loads(output_item.arguments)
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
                        part.text
                        for part in output_item.content
                        if part.type == "output_text"
                    )
                    print(text)
                    if output_item.status == "completed":
                        return text
                case "reasoning":
                    continue
                case _:
                    print("Unhandled type")
                    print(output_item.type)
                    print(output_item)

        prior_output = [
            cast(ResponseInputItemParam, item.model_dump(exclude_none=True))
            for item in output
        ]
        next_input = [*old_input, *prior_output, *append_to_input]
        response = await self.call_model(next_input)
        return await self.handle_response(next_input, response.output, cwd)

    async def call_model(
        self, prompt: str | list[ResponseInputItemParam]
    ) -> Response:
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
            print("TOOL CALL: run shell command")
            print(args["cmd"])
            return await self.run_shell_command(args["cmd"], cwd)
        raise ValueError(f"Unknown function: {name}")

    async def run_shell_command(self, cmd: str, cwd: Path | None = None) -> str:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(cwd) if cwd is not None else None,
        )
        return str(result)
