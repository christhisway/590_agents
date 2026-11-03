"""Exercise the webp_leonscottkennedy loop with a scripted client and real local shell commands."""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from openai import APIConnectionError, AsyncOpenAI
from openai.types.responses import Response

from agents590 import atif
from agents590.contracts import RunContext
from agents590.webp_leonscottkennedy import (
    CONTINUE_PROMPT,
    SELF_TEST_SCRIPT,
    LeonScottKennedy,
    run_shell,
)


def _response(
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
    fields: dict[str, Any] | None = None,
) -> Response:
    return Response.model_validate(
        {
            **(fields or {}),
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "model": "course-deployment",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": usage,
        }
    )


def _usage(prompt: int, completion: int, cached: int) -> dict[str, Any]:
    return {
        "input_tokens": prompt,
        "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 0},
        "output_tokens": completion,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": prompt + completion,
    }


def _shell_call(call_id: str, command: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": "shell",
        "arguments": json.dumps({"command": command}),
        "status": "completed",
    }


def _message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


class ScriptedResponses:
    def __init__(self, responses: list[Response | Exception]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: object) -> Response:
        self.requests.append(json.loads(json.dumps(kwargs)))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class ScriptedClient:
    def __init__(self, responses: list[Response | Exception]) -> None:
        self.responses = ScriptedResponses(responses)


@pytest.fixture(autouse=True)
def configured_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unc-key")
    monkeypatch.setenv("OPENAI_MODEL", "course-deployment")


def test_loop_runs_shell_calls_in_cwd_and_returns_final_text(tmp_path: Path) -> None:
    client = ScriptedClient(
        [
            _response([_shell_call("call_1", "echo fixed > result.txt && pwd")]),
            _response([_message("Done.")]),
        ]
    )
    agent = LeonScottKennedy(client=cast(AsyncOpenAI, client))
    output = asyncio.run(agent.run("Fix it", RunContext(cwd=tmp_path, model="openai/other")))

    assert output == "Done."
    assert (tmp_path / "result.txt").read_text() == "fixed\n"
    first, second = client.responses.requests
    assert first["model"] == "other"
    assert first["store"] is False and first["background"] is False
    assert first["input"] == [{"role": "user", "content": "Fix it"}]
    tool_result = second["input"][-1]
    assert tool_result["type"] == "function_call_output"
    assert tool_result["call_id"] == "call_1"
    assert "exit_code: 0" in tool_result["output"]
    assert str(tmp_path) in tool_result["output"]


@pytest.mark.parametrize("conventions", [None, "Run `make check` after every change."])
def test_system_prompt_includes_agents_md_when_present(
    tmp_path: Path, conventions: str | None
) -> None:
    if conventions is not None:
        (tmp_path / "AGENTS.md").write_text(f"# Conventions\n\n{conventions}\n")
    client = ScriptedClient([_response([_message("ok")])])
    asyncio.run(
        LeonScottKennedy(client=cast(AsyncOpenAI, client)).run("x", RunContext(cwd=tmp_path))
    )

    instructions = client.responses.requests[0]["instructions"]
    assert str(tmp_path) in instructions
    if conventions is None:
        assert "AGENTS.md" not in instructions
    else:
        assert instructions.endswith(f"# Conventions\n\n{conventions}")


@pytest.mark.parametrize("contents", ["", "\n  \t\n"])
def test_empty_agents_md_adds_nothing_to_system_prompt(tmp_path: Path, contents: str) -> None:
    baseline_client = ScriptedClient([_response([_message("ok")])])
    asyncio.run(
        LeonScottKennedy(client=cast(AsyncOpenAI, baseline_client)).run(
            "x", RunContext(cwd=tmp_path)
        )
    )
    baseline = baseline_client.responses.requests[0]["instructions"]

    (tmp_path / "AGENTS.md").write_text(contents)
    client = ScriptedClient([_response([_message("ok")])])
    asyncio.run(
        LeonScottKennedy(client=cast(AsyncOpenAI, client)).run("x", RunContext(cwd=tmp_path))
    )

    assert client.responses.requests[0]["instructions"] == baseline


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        ({**_shell_call("c", ""), "arguments": "not json"}, "expected a JSON object"),
        ({**_shell_call("c", ""), "arguments": "[]"}, "expected a JSON object"),
        ({**_shell_call("c", ""), "arguments": '{"command": 1}'}, "must be a string"),
        ({**_shell_call("c", ""), "name": "python"}, "available tools: shell"),
    ],
)
def test_malformed_tool_calls_are_reported_to_the_model(
    tmp_path: Path, call: dict[str, Any], expected: str
) -> None:
    client = ScriptedClient([_response([call]), _response([_message("ok")])])
    asyncio.run(
        LeonScottKennedy(client=cast(AsyncOpenAI, client)).run("x", RunContext(cwd=tmp_path))
    )
    assert expected in client.responses.requests[1]["input"][-1]["output"]


def test_run_shell_reports_failures_and_timeouts(tmp_path: Path) -> None:
    failed = asyncio.run(run_shell("echo oops >&2; exit 3", tmp_path))
    assert "exit_code: 3" in failed and "oops" in failed
    assert "timed out" in asyncio.run(run_shell("sleep 5", tmp_path, timeout=0.1))


def test_run_shell_timeout_kills_child_process(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    result = asyncio.run(
        run_shell(f"sleep 30 & child=$!; echo $child > {pid_path}; wait", tmp_path, timeout=0.1)
    )
    assert "timed out" in result

    child_pid = int(pid_path.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("child process survived the shell timeout")


def test_each_iteration_appends_an_atif_step(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    summary = [{"type": "summary_text", "text": "Look first."}]
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": summary}
    malformed = _shell_call("call_2", "")
    client = ScriptedClient(
        [
            _response([reasoning, _shell_call("call_1", "echo hi")], _usage(100, 20, 40)),
            _response([{**malformed, "arguments": "not json"}], _usage(150, 10, 100)),
            _response([_message("Done.")], _usage(200, 5, 150)),
        ]
    )
    agent = LeonScottKennedy(client=cast(AsyncOpenAI, client), trajectory_path=path)
    asyncio.run(agent.run("Fix it", RunContext(cwd=tmp_path)))

    trajectory = json.loads(path.read_text())
    steps = trajectory["steps"]
    assert trajectory["schema_version"] == "ATIF-v1.8"
    assert trajectory["agent"]["name"] == "webp_leonscottkennedy"
    assert [step["step_id"] for step in steps] == [1, 2, 3, 4, 5]
    assert [step["source"] for step in steps] == ["system", "user", "agent", "agent", "agent"]
    assert steps[1]["message"] == "Fix it"
    first = steps[2]
    assert first["reasoning_content"] == "Look first."
    assert first["tool_calls"] == [
        {"tool_call_id": "call_1", "function_name": "shell", "arguments": {"command": "echo hi"}}
    ]
    assert first["observation"]["results"][0]["source_call_id"] == "call_1"
    assert "hi" in first["observation"]["results"][0]["content"]
    assert steps[3]["tool_calls"][0]["arguments"] == {"raw": "not json"}
    assert steps[4]["message"] == "Done." and "tool_calls" not in steps[4]
    assert trajectory["final_metrics"] == {
        "total_prompt_tokens": 450,
        "total_completion_tokens": 35,
        "total_cached_tokens": 290,
        "total_steps": 5,
    }
    assert not (tmp_path / "trajectory.tmp").exists()


def test_trajectory_is_only_written_inside_harbor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run_once() -> None:
        client = ScriptedClient([_response([_message("ok")])])
        agent = LeonScottKennedy(client=cast(AsyncOpenAI, client))
        asyncio.run(agent.run("x", RunContext(cwd=tmp_path)))

    monkeypatch.delenv("AGENTS590_ATIF_PATH", raising=False)
    monkeypatch.setattr(atif, "HARBOR_AGENT_LOGS", tmp_path / "missing")
    run_once()
    assert list(tmp_path.iterdir()) == []

    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(atif, "HARBOR_AGENT_LOGS", logs)
    run_once()
    assert (logs / "trajectory.json").is_file()

    override = tmp_path / "override.json"
    monkeypatch.setenv("AGENTS590_ATIF_PATH", str(override))
    run_once()
    assert override.is_file()


def test_unwritable_trajectory_stops_the_run(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    client = ScriptedClient([_response([_message("Done.")])])
    agent = LeonScottKennedy(client=cast(AsyncOpenAI, client), trajectory_path=blocker / "t.json")

    with pytest.raises(OSError):
        asyncio.run(agent.run("Fix it", RunContext(cwd=tmp_path)))
    assert client.responses.requests == []


@pytest.mark.parametrize("own_repo", [False, True])
def test_system_prompt_points_at_self_test_script_only_in_own_repo(
    tmp_path: Path, own_repo: bool
) -> None:
    if own_repo:
        (tmp_path / SELF_TEST_SCRIPT).parent.mkdir(parents=True)
        (tmp_path / SELF_TEST_SCRIPT).write_text("")
    client = ScriptedClient([_response([_message("ok")])])
    agent = LeonScottKennedy(client=cast(AsyncOpenAI, client))
    asyncio.run(agent.run("x", RunContext(cwd=tmp_path)))

    assert (str(SELF_TEST_SCRIPT) in client.responses.requests[0]["instructions"]) == own_repo


def test_api_errors_end_the_run_with_a_recorded_reason(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    error = APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
    client = ScriptedClient([_response([_shell_call("call_1", "echo hi")]), error])
    agent = LeonScottKennedy(client=cast(AsyncOpenAI, client), trajectory_path=path)
    output = asyncio.run(agent.run("x", RunContext(cwd=tmp_path)))

    assert output.startswith("Stopped early: the model request failed (APIConnectionError")
    last = json.loads(path.read_text())["steps"][-1]
    assert last["source"] == "system" and last["message"] == output


def test_output_limit_cut_off_asks_the_model_to_continue(tmp_path: Path) -> None:
    cut_off = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
    client = ScriptedClient(
        [_response([_message("Half an ans")], fields=cut_off), _response([_message("Done.")])]
    )
    agent = LeonScottKennedy(client=cast(AsyncOpenAI, client))
    output = asyncio.run(agent.run("x", RunContext(cwd=tmp_path)))

    assert output == "Done."
    assert client.responses.requests[1]["input"][-1] == {"role": "user", "content": CONTINUE_PROMPT}


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        (
            {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}},
            "incomplete (content_filter)",
        ),
        (
            {"status": "failed", "error": {"code": "server_error", "message": "boom"}},
            "failed (boom)",
        ),
    ],
)
def test_failed_or_filtered_responses_stop_without_running_tools(
    tmp_path: Path, fields: dict[str, Any], expected: str
) -> None:
    client = ScriptedClient([_response([_shell_call("call_1", "touch ran")], fields=fields)])
    agent = LeonScottKennedy(client=cast(AsyncOpenAI, client))
    output = asyncio.run(agent.run("x", RunContext(cwd=tmp_path)))

    assert expected in output
    assert not (tmp_path / "ran").exists()


def test_step_limit_comes_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENTS590_MAX_STEPS", "1")
    client = ScriptedClient([_response([_shell_call("call_1", "echo hi")])])

    output = asyncio.run(
        LeonScottKennedy(client=cast(AsyncOpenAI, client)).run("x", RunContext(cwd=tmp_path))
    )

    assert output == "Stopped after reaching the 1-step limit."
    assert len(client.responses.requests) == 1


@pytest.mark.parametrize("value", ["", "0", "-1", "one", "1.5"])
def test_invalid_step_limit_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str
) -> None:
    monkeypatch.setenv("AGENTS590_MAX_STEPS", value)
    client = ScriptedClient([])

    with pytest.raises(ValueError, match="AGENTS590_MAX_STEPS must be a positive integer"):
        asyncio.run(
            LeonScottKennedy(client=cast(AsyncOpenAI, client)).run("x", RunContext(cwd=tmp_path))
        )
    assert client.responses.requests == []
