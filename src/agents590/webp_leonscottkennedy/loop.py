"""One request/tool-call iteration and trajectory recording."""

from pathlib import Path
from typing import Any, cast

from openai import APIError, AsyncOpenAI
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseInputItemParam,
    ResponseOutputItem,
    ResponseReasoningItem,
)

from agents590.atif import ATIFRecorder, JSONValue
from agents590.contracts import RunContext

from .prompts import CONTINUE_PROMPT, system_prompt
from .tools import Tool, call_tool, tools_for

REQUEST_TIMEOUT_SECONDS = 300.0


async def request(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    tools: dict[str, Tool],
    transcript: list[ResponseInputItemParam],
) -> Response:
    return await client.responses.create(
        model=model,
        instructions=system_prompt,
        input=transcript,
        tools=[tool.schema for tool in tools.values()],
        include=["reasoning.encrypted_content"],
        store=False,
        background=False,
    )


async def run_calls(
    tools: dict[str, Tool], calls: list[ResponseFunctionToolCall], cwd: Path
) -> list[tuple[str, str]]:
    return [
        (call.call_id, await call_tool(tools, call.name, call.arguments, cwd)) for call in calls
    ]


def as_input(item: ResponseOutputItem) -> ResponseInputItemParam:
    return cast(ResponseInputItemParam, item.model_dump(exclude_none=True))


def call_output(call_id: str, output: str) -> ResponseInputItemParam:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _arguments_object(arguments: str) -> dict[str, Any]:
    import json

    try:
        parsed: object = json.loads(arguments)
    except json.JSONDecodeError:
        parsed = None
    return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else {"raw": arguments}


def record_response(
    recorder: ATIFRecorder, response: Response, outputs: list[tuple[str, str]]
) -> None:
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


def stop_reason(response: Response) -> str | None:
    if response.status == "failed":
        detail = response.error.message if response.error is not None else "no details"
        return f"the model response failed ({detail})"
    if response.status == "incomplete":
        details = response.incomplete_details
        reason = details.reason if details is not None else None
        if reason != "max_output_tokens":
            return f"the model response was incomplete ({reason or 'unknown reason'})"
    return None


def stop(recorder: ATIFRecorder, reason: str) -> str:
    message = f"Stopped early: {reason}."
    recorder.add("system", message)
    return message


async def execute_loop(
    recorder: ATIFRecorder,
    client: AsyncOpenAI,
    model: str,
    instruction: str,
    context: RunContext,
    max_steps: int,
) -> str:
    from . import skills

    found = skills.discover(context.cwd)
    system = system_prompt(context.cwd, found)
    tools = tools_for(found)
    recorder.add("system", system)
    recorder.add("user", instruction)
    transcript: list[ResponseInputItemParam] = [{"role": "user", "content": instruction}]
    for _ in range(max_steps):
        try:
            response = await request(client, model, system, tools, transcript)
        except APIError as error:
            return stop(recorder, f"the model request failed ({type(error).__name__}: {error})")
        if (reason := stop_reason(response)) is not None:
            record_response(recorder, response, [])
            return stop(recorder, reason)
        calls = [item for item in response.output if isinstance(item, ResponseFunctionToolCall)]
        outputs = await run_calls(tools, calls, context.cwd)
        transcript += [as_input(item) for item in response.output]
        transcript += [call_output(call_id, output) for call_id, output in outputs]
        record_response(recorder, response, outputs)
        if not calls and response.status == "incomplete":
            transcript.append({"role": "user", "content": CONTINUE_PROMPT})
            recorder.add("user", CONTINUE_PROMPT)
        elif not calls:
            return response.output_text
    return f"Stopped after reaching the {max_steps}-step limit."
