"""Build the instructions supplied to the coding agent."""

import os
from pathlib import Path

from . import skills

MAX_STEPS = 50
MAX_STEPS_ENV = "AGENTS590_MAX_STEPS"
MAX_OUTPUT_CHARS = 20_000
PROJECT_INSTRUCTIONS_FILE = "AGENTS.md"
SELF_TEST_SCRIPT = Path("src/agents590/webp_leonscottkennedy/test.sh")
SELF_TEST_NOTE = """

This repository contains your own source code. To run the tests, use `{script}` in place \
of `make test`; it also runs your own tests, which `make test` skips."""
CONTINUE_PROMPT = "Your previous response was cut off at the output limit. Continue the task."
SYSTEM_PROMPT = """\
You are an autonomous coding agent working in a Linux or macOS shell. Nobody will \
answer questions, so complete the task end to end without asking for confirmation.

Use the `shell` tool to inspect files, edit them, and run code or tests. Each call \
runs in a fresh `bash -c` with the working directory {cwd}; state such as `cd` does \
not persist between calls. Prefer small, precise edits (for example a Python script, \
`sed`, or a heredoc that rewrites a whole file you have just read). Change only what \
the task requires and preserve unrelated content exactly.

Before finishing, verify your change by running it or its tests where possible. \
When the task is complete, reply with a short summary and no tool call."""


def configured_max_steps() -> int:
    value = os.environ.get(MAX_STEPS_ENV)
    if value is None:
        return MAX_STEPS
    if not value.isdecimal() or int(value) <= 0:
        raise ValueError(f"{MAX_STEPS_ENV} must be a positive integer; got {value!r}.")
    return int(value)


def truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    omitted = len(text) - MAX_OUTPUT_CHARS
    return f"{text[:half]}\n... [{omitted} characters omitted] ...\n{text[-half:]}"


def system_prompt(cwd: Path, found: list[skills.Skill]) -> str:
    prompt = SYSTEM_PROMPT.format(cwd=cwd)
    if (cwd / SELF_TEST_SCRIPT).is_file():
        prompt += SELF_TEST_NOTE.format(script=SELF_TEST_SCRIPT)
    prompt += skills.catalog(found)
    path = cwd / PROJECT_INSTRUCTIONS_FILE
    if not path.is_file():
        return prompt
    instructions = truncate(path.read_text(errors="replace").strip())
    if not instructions:
        return prompt
    return (
        f"{prompt}\n\nThe project's {PROJECT_INSTRUCTIONS_FILE} follows. "
        f"Treat it as standing instructions for every change you make.\n\n{instructions}"
    )
