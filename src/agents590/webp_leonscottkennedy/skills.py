"""Agent Skills (https://agentskills.io): find SKILL.md files and load them on demand.

The catalog in the system prompt gives only each skill's name and description. The model
activates a skill by name to receive its instructions and a list of its bundled files,
which it then reads with the shell only when the instructions call for them.
"""

import re
from dataclasses import dataclass
from html import escape
from pathlib import Path

# Scanned in order; a later skill replaces an earlier one with the same name, so a
# project's skills override the ones bundled with this agent.
BUNDLED_SKILLS = Path(__file__).parent / "skills"
PROJECT_SKILLS = Path(".agents/skills")

CATALOG_INSTRUCTIONS = """

The following skills provide specialized instructions for specific tasks. When a task \
matches a skill's description, call `activate_skill` with its name to load its full \
instructions before proceeding."""

# Bundled files listed on activation; the model reads the ones it needs with the shell.
MAX_RESOURCES = 50

FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*$", re.DOTALL | re.MULTILINE)
FIELD = re.compile(r"([A-Za-z][\w-]*):(.*)")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    location: Path


def _scalar(value: str) -> str:
    """Drop a block-scalar indicator or surrounding quotes from a YAML value."""
    value = re.sub(r"\A[>|][+-]?\s*", "", value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _frontmatter(text: str) -> dict[str, str]:
    """Read top-level fields, folding indented lines into the field above them.

    A deliberately lenient subset of YAML: values may contain colons, as many skills
    written for other clients do, and nested maps such as `metadata` are left unparsed.
    """
    match = FRONTMATTER.match(text)
    fields: dict[str, str] = {}
    key = None
    for line in match.group(1).splitlines() if match else []:
        if field := FIELD.fullmatch(line):
            key, fields[field[1]] = field[1], field[2]
        elif key is not None and line.strip():
            fields[key] += f" {line.strip()}"
    return {key: _scalar(value) for key, value in fields.items()}


def _load(location: Path) -> Skill | None:
    """Parse one SKILL.md, skipping it when unreadable or without a description."""
    try:
        fields = _frontmatter(location.read_text(errors="replace"))
    except OSError:
        return None
    if not (description := fields.get("description")):
        return None
    name = fields.get("name") or location.parent.name
    return Skill(name, description, location.resolve())


def discover(cwd: Path) -> list[Skill]:
    """Find the bundled skills and the project's, keyed by name so the project's win."""
    found: dict[str, Skill] = {}
    for root in (BUNDLED_SKILLS, cwd / PROJECT_SKILLS):
        for location in sorted(root.glob("*/SKILL.md")):
            if skill := _load(location):
                found[skill.name] = skill
    return list(found.values())


def catalog(skills: list[Skill]) -> str:
    """Describe the skills for the system prompt, or nothing when there are none."""
    if not skills:
        return ""
    entries = "".join(
        f"\n  <skill>\n    <name>{escape(skill.name)}</name>"
        f"\n    <description>{escape(skill.description)}</description>\n  </skill>"
        for skill in skills
    )
    return f"{CATALOG_INSTRUCTIONS}\n\n<available_skills>{entries}\n</available_skills>"


def activate(skill: Skill) -> str:
    """Return the skill's instructions without frontmatter, its directory, and its files."""
    try:
        text = skill.location.read_text(errors="replace")
    except OSError as error:
        return f"Could not read skill {skill.name!r}: {error}"
    match = FRONTMATTER.match(text)
    body = text[match.end() :] if match else text
    directory = skill.location.parent
    files = sorted(
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path != skill.location
    )
    listing = "".join(f"\n  <file>{escape(file)}</file>" for file in files[:MAX_RESOURCES])
    if len(files) > MAX_RESOURCES:
        listing += f"\n  <!-- {len(files) - MAX_RESOURCES} more files not listed -->"
    resources = f"\n\n<skill_resources>{listing}\n</skill_resources>" if files else ""
    return (
        f'<skill_content name="{escape(skill.name)}">\n{body.strip()}\n\n'
        f"Skill directory: {directory}\n"
        f"Relative paths in this skill are relative to the skill directory."
        f"{resources}\n</skill_content>"
    )
