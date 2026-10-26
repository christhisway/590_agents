"""Small dependency-free ATIF writer used by agents running under Harbor."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import NotRequired, TypedDict, cast

# ATIF permits additional JSON members on steps.  Keeping the JSON value type
# here prevents the recorder's public data model from falling back to ``Any``.
type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]


class ATIFAgent(TypedDict):
    name: str
    version: str
    model_name: str


class ATIFMetrics(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int


class ATIFToolCall(TypedDict):
    tool_call_id: str
    function_name: str
    arguments: JSONValue


class ATIFObservationResult(TypedDict):
    source_call_id: str
    content: str


class ATIFObservation(TypedDict):
    results: list[ATIFObservationResult]


class ATIFFinalMetrics(TypedDict):
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cached_tokens: int
    total_steps: int


class ATIFStep(TypedDict):
    step_id: int
    timestamp: str
    source: str
    message: str
    model_name: NotRequired[str]
    tool_calls: NotRequired[list[ATIFToolCall]]
    metrics: NotRequired[ATIFMetrics]
    observation: NotRequired[ATIFObservation]


class ATIFDocument(TypedDict):
    schema_version: str
    session_id: str
    agent: ATIFAgent
    steps: list[ATIFStep]
    final_metrics: NotRequired[ATIFFinalMetrics]


class ATIFRecorder:
    """Collect a Responses conversation and write an ATIF trajectory on each update."""

    def __init__(self, path: Path | None, *, agent: str, model: str) -> None:
        self.path = path
        self.document: ATIFDocument = {
            "schema_version": "ATIF-v1.8",
            "session_id": str(uuid.uuid4()),
            "agent": {"name": agent, "version": "0.1.0", "model_name": model},
            "steps": [],
        }

    @classmethod
    def from_environment(cls, *, agent: str, model: str) -> ATIFRecorder:
        configured = os.environ.get("AGENTS590_ATIF_PATH", "").strip()
        # Harbor exposes /agent inside installed-agent containers. Keep
        # ordinary local CLI runs side-effect free when that directory is absent.
        path = (
            Path(configured)
            if configured
            else (Path("/agent/trajectory.json") if Path("/agent").is_dir() else None)
        )
        return cls(path, agent=agent, model=model)

    def add(self, source: str, message: str, **fields: JSONValue) -> None:
        step: dict[str, JSONValue] = {
            "step_id": len(self.document["steps"]) + 1,
            "timestamp": datetime.now(UTC).isoformat(),
            "source": source,
            "message": message,
        }
        # ATIF defines extension fields, so only the known fields above are
        # required in the TypedDict; extension values are still JSON-typed.
        step.update(fields)
        self.document["steps"].append(cast(ATIFStep, step))
        self.write()

    def write(self) -> None:
        """Atomically write the latest valid trajectory, if recording is enabled."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.document, indent=2) + "\n")
        temporary.replace(self.path)
