"""Public agent implementation."""

from pathlib import Path

from openai import AsyncOpenAI

from agents590.atif import ATIFRecorder
from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings

from .loop import REQUEST_TIMEOUT_SECONDS, execute_loop
from .prompts import configured_max_steps


class LeonScottKennedy:
    """Loop between the model and its tools until the model stops calling them."""

    def __init__(
        self, client: AsyncOpenAI | None = None, trajectory_path: Path | None = None
    ) -> None:
        self._client = client
        self._trajectory_path = trajectory_path

    async def run(self, instruction: str, context: RunContext) -> str:
        max_steps = configured_max_steps()
        settings = UNCOpenAISettings.from_env(model=context.model)
        if self._client is not None:
            return await execute_loop(
                self._recorder(settings.model),
                self._client,
                settings.model,
                instruction,
                context,
                max_steps,
            )
        async with settings.create_async_client() as client:
            patient = client.with_options(timeout=REQUEST_TIMEOUT_SECONDS, max_retries=2)
            return await execute_loop(
                self._recorder(settings.model),
                patient,
                settings.model,
                instruction,
                context,
                max_steps,
            )

    def _recorder(self, model: str) -> ATIFRecorder:
        if self._trajectory_path is not None:
            return ATIFRecorder(self._trajectory_path, agent="webp_leonscottkennedy", model=model)
        return ATIFRecorder.from_environment(agent="webp_leonscottkennedy", model=model)
