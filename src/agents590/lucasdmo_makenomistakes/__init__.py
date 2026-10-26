from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from collections.abc import Iterable, Mapping
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

from agents590.atif import (
    ATIFFinalMetrics,
    ATIFMetrics,
    ATIFRecorder,
    ATIFToolCall,
    JSONValue,
)
from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings

logger = logging.getLogger(__name__)


class MyAgent:
    """Scaffold for a model/tool loop using the shared standalone interface."""

    client: AsyncOpenAI
    settings: UNCOpenAISettings
    orig_prompt: str
    recorder: ATIFRecorder

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
        self.recorder = ATIFRecorder.from_environment(
            agent=type(self).__name__, model=self.settings.model
        )
        self.recorder.add("user", instruction)
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
                    if self.recorder.document["steps"]:
                        step = self.recorder.document["steps"][-1]
                        observation = step.setdefault("observation", {"results": []})
                        observation["results"].append(
                            {"source_call_id": output_item.call_id, "content": result}
                        )
                        self.recorder.write()
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

    def record_atif_response(self, response: Response) -> None:
        """Record one model response in ATIF and update cumulative metrics.

        Keeping this piping separate from ``handle_response`` makes the tool
        loop responsible only for interpreting model output.  It also ensures
        that reasoning-only responses are recorded, since they may contain
        usage without producing user-visible text.
        """
        output = list(response.output)
        tool_calls: list[ATIFToolCall] = []
        text_parts: list[str] = []
        for output_item in output:
            if output_item.type == "function_call":
                try:
                    args_for_record = json.loads(output_item.arguments)
                except json.JSONDecodeError:
                    args_for_record = {"raw_arguments": output_item.arguments}
                tool_calls.append(
                    {
                        "tool_call_id": output_item.call_id,
                        "function_name": output_item.name,
                        "arguments": cast(JSONValue, args_for_record),
                    }
                )
            elif output_item.type == "message":
                text_parts.extend(
                    part.text for part in output_item.content if part.type == "output_text"
                )

        metrics = self._metrics(getattr(response, "usage", None))
        if tool_calls or text_parts or metrics is not None:
            fields: dict[str, JSONValue] = {"model_name": self.settings.model}
            if tool_calls:
                fields["tool_calls"] = cast(JSONValue, tool_calls)
            if metrics is not None:
                fields["metrics"] = cast(JSONValue, metrics)
            self.recorder.add("agent", "\n".join(text_parts), **fields)

        if metrics is not None:
            self._update_final_metrics()
            logger.info(
                "ATIF token usage: response_id=%s model=%s prompt_tokens=%d "
                "completion_tokens=%d cached_tokens=%d trajectory=%s",
                getattr(response, "id", None),
                self.settings.model,
                metrics["prompt_tokens"],
                metrics["completion_tokens"],
                metrics["cached_tokens"],
                self.recorder.path or "disabled",
            )
        else:
            logger.warning(
                "Responses API returned no usage; ATIF metrics were not recorded: "
                "response_id=%s model=%s trajectory=%s",
                getattr(response, "id", None),
                self.settings.model,
                self.recorder.path or "disabled",
            )

    @staticmethod
    def _metrics(usage: object) -> ATIFMetrics | None:
        """Translate Responses usage into the ATIF token accounting fields.

        ``input_tokens`` is already the complete prompt total.  Cached input
        is a subset of it, so it must not be subtracted before writing the
        ATIF step (or the dashboard will under-report prompt size).
        """
        if usage is None:
            return None

        def value(obj: object, name: str, default: object = 0) -> object:
            if isinstance(obj, dict):
                return cast(Mapping[str, object], obj).get(name, default)
            return getattr(obj, name, default)

        def integer(value_to_convert: object) -> int:
            if isinstance(value_to_convert, int):
                return value_to_convert
            if isinstance(value_to_convert, str):
                try:
                    return int(value_to_convert)
                except ValueError:
                    pass
            return 0

        input_tokens = integer(value(usage, "input_tokens"))
        output_tokens = integer(value(usage, "output_tokens"))
        details = value(usage, "input_tokens_details", None)
        cached_tokens = integer(value(details, "cached_tokens")) if details else 0

        # Keep malformed provider data from producing invalid ATIF metrics.
        prompt_tokens = max(0, input_tokens)
        completion_tokens = max(0, output_tokens)
        cached_tokens = max(0, min(cached_tokens, prompt_tokens))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
        }

    def _update_final_metrics(self) -> None:
        """Publish cumulative token totals for consumers that use the summary."""
        totals: dict[str, int] = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cached_tokens": 0,
        }
        for step in self.recorder.document["steps"]:
            metrics = step.get("metrics")
            if not metrics:
                continue
            totals["total_prompt_tokens"] += int(metrics.get("prompt_tokens", 0))
            totals["total_completion_tokens"] += int(metrics.get("completion_tokens", 0))
            totals["total_cached_tokens"] += int(metrics.get("cached_tokens", 0))
        totals["total_steps"] = len(self.recorder.document["steps"])
        self.recorder.document["final_metrics"] = cast(ATIFFinalMetrics, totals)
        self.recorder.write()

    async def call_model(self, prompt: str | list[ResponseInputItemParam]) -> Response:
        response = await self.client.responses.create(
            model=self.settings.model,
            instructions="ORIGINAL USER PROMPT: " + self.orig_prompt,
            tools=self.tools,
            input=prompt,
            store=False,
            background=False,
        )
        self.last_response = response
        self.record_atif_response(response)
        return response

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
