"""The course starter agent, independent of Harbor."""

from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings
from openai import AsyncOpenAI
from typing import Any
import json
import subprocess


class MyAgent:
    """Scaffold for a model/tool loop using the shared standalone interface."""

    client: AsyncOpenAI
    settings: UNCOpenAISettings
    orig_prompt: str

    def provider_settings(self, context: RunContext) -> UNCOpenAISettings:
        """Prefer the per-run model to the environment's configured deployment."""
        return UNCOpenAISettings.from_env(model=context.model)

    tools = [
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
            }
        },
        # {
        #     "type": "function",
        #     "name": "eval_python",
        #     "description": "Run arbitrary Python code via the eval() function. This has the same limitations as eval()",
        #     "parameters": {
        #     "type": "object",
        #     "properties": {
        #         "code": {
        #             "type": "string",
        #             "description": "Code to execute, ex. 2+2",
        #         },
        #     },
        #     "required": ["code"],
        #     }
        # },
        # {
        #     "type": "function",
        #     "name": "exec_python",
        #     "description": "Run arbitrary Python code via the exec() function.",
        #     "parameters": {
        #     "type": "object",
        #          "properties": {
        #             "code": {
        #                 "type": "string",
        #                 "description": "Code to execute, ex. 2+2",
        #             },
        #         },
        #         "required": ["code"],
        #     }
        # }
    ]

    async def run(self, instruction: str, context: RunContext) -> str:
        # Start with settings = self.provider_settings(context), then open
        # settings.create_async_client() and use model=settings.model.
        # Send Responses requests with store=False and background=False.
        # Run shell tools with context.cwd as their explicit working directory.
        
        self.settings = self.provider_settings(context)
        self.client = self.settings.create_async_client()
        self.orig_prompt = instruction
        input_messages: list[str] = [instruction]

        response = await self.call_model("\n".join(input_messages))
        await self.handle_response(input_messages, response.output)

    async def handle_response(self, old_input: str, output: str) -> list[str]:
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
        if complete is True:
            return []
        old_input.insert(0, "ORIGINAL PROMPT: " + self.orig_prompt)
        response = await self.call_model("\n".join(old_input + append_to_input))
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
        if name == "run_shell_command":
            print("TOOL CALL: run shell command")
            print(args["cmd"])
            return await self.run_shell_command(**args)
        if name == "eval_python":
            return await self.eval_python(**args)
        elif name == "exec_python":
            return await self.exec_python(**args)
        raise ValueError(f"Unknown function: {name}")

    # async def eval_python(self, code: str) -> str:
    #     return eval(code)

    # async def exec_python(self, code: str) -> str:
    #         return exec(code)

    async def run_shell_command(self, cmd: str) -> str:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=False
        )
        return str(result)
