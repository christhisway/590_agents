"""The course starter agent, independent of Harbor."""

from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings
from openai import AsyncOpenAI, BadRequestError
from typing import Any
import json


class MyAgent:
    """Scaffold for a model/tool loop using the shared standalone interface."""

    client: AsyncOpenAI
    settings: UNCOpenAISettings

    def provider_settings(self, context: RunContext) -> UNCOpenAISettings:
        """Prefer the per-run model to the environment's configured deployment."""
        return UNCOpenAISettings.from_env(model=context.model)

    tools = [
        {
            "type": "function",
            "name": "eval_python",
            "description": "Run arbitrary Python code via eval()",
            "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Code to execute, ex. 2+2",
                },
            },
            "required": ["code"],
            }
        }
    ]

    async def run(self, instruction: str, context: RunContext) -> str:
        # Start with settings = self.provider_settings(context), then open
        # settings.create_async_client() and use model=settings.model.
        # Send Responses requests with store=False and background=False.
        # Run shell tools with context.cwd as their explicit working directory.
        self.settings = self.provider_settings(context)
        self.client = self.settings.create_async_client()

        input_messages: list[str] = [instruction]

        response = await self.call_model("\n".join(input_messages))
        await self.handle_response(input_messages, response.output)

    async def handle_response(self, old_input, output) -> list[str]:
        append_to_input: list[str] = []
        complete = False
        for output_item in output:
            match output_item.type:
                case "function_call":
                    name = output_item.name
                    args = json.loads(output_item.arguments)
        
                    result = await self.call_function(name, args)
                    append_to_input.append(
                        str({
                            "type": "function_call_output",
                            "call_id": output_item.call_id,
                            "output": json.dumps(result),
                        })
                    )
                case "message":
                    print(output_item.content[0].text)
                    if output_item.status is "completed":
                        complete = True
                case "reasoning":
                    # ignore
                    continue
                case _:
                    print("Unhandled type")
                    print(output_item.type)
                    print(output_item)
        response = await self.call_model("\n".join(old_input + append_to_input))
        if complete is True:
            return []
        return await self.handle_response(append_to_input, response.output)

    async def call_model(self, prompt: str):
        response = await self.client.responses.create(
            model=self.settings.model,
            instructions="",
            tools=self.tools,
            input=prompt,
            store=False,
            background=False,
        )
        return response

    async def call_function(self, name: str, args: Any) -> Any:
        if name == "eval_python":
            return await self.eval_python(**args)
        raise ValueError(f"Unknown function: {name}")

    async def eval_python(self, code: str) -> str:
        return eval(code)
