---
name: create-skill
description: Write a new Agent Skill (a directory with a SKILL.md) or improve an existing one. Use when asked to create, add, edit, or review a skill, or to capture a repeatable workflow as reusable instructions.
---

# Create a skill

A skill is a directory whose `SKILL.md` holds YAML frontmatter and Markdown instructions.
Agents see only each skill's `name` and `description` until a task matches, then read the
whole `SKILL.md`, then open bundled files only when the instructions point to them.

## Where it goes

- In a project, put it at `.agents/skills/<name>/SKILL.md` unless told otherwise.
- The directory name must equal the `name` field.

## Frontmatter

```markdown
---
name: pdf-processing
description: Extract text and tables from PDFs, fill forms, and merge files. Use when working with PDF documents or when the user mentions PDFs or forms.
---
```

- `name` (required): 1-64 characters of `a-z`, `0-9`, and `-`; no leading, trailing,
  or consecutive hyphens.
- `description` (required): at most 1024 characters. Say what the skill does and when to
  use it, with the keywords a matching task would contain. This is all an agent sees when
  deciding whether to load the skill, so make it specific.
- Optional: `license`, `compatibility` (environment requirements, at most 500
  characters), `metadata` (a map of string keys to string values), `allowed-tools`.
- Quote a value that contains `: ` so it stays valid YAML.

## Body

- Write the steps an agent should follow, with a short example and the edge cases that
  trip people up. Skip what a capable agent already knows.
- Keep `SKILL.md` under 500 lines. Move long material into `references/`, runnable code
  into `scripts/`, and templates or data into `assets/`.
- Link bundled files by paths relative to the skill directory, one level deep, such as
  `scripts/extract.py` or `references/REFERENCE.md`, and say when to open each one.

## Check it

1. The frontmatter opens on the first line with `---` and closes with `---`.
2. `name` matches the directory and the naming rules above.
3. Every file the body references exists, and every bundled script runs.
