"""Check skill discovery, parsing, and the catalog the agent puts in its system prompt."""

from pathlib import Path

import pytest

from agents590.webp_leonscottkennedy import skills


def _write_skill(root: Path, directory: str, frontmatter: str, body: str = "# Steps") -> Path:
    path = root / directory / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(f"---\n{frontmatter}\n---\n\n{body}\n")
    return path


def test_bundled_default_skill_is_valid() -> None:
    (skill,) = skills.discover(Path("/nonexistent"))
    assert skill.name == "create-skill" == skill.location.parent.name
    assert skill.location == skills.BUNDLED_SKILLS / "create-skill" / "SKILL.md"
    assert 0 < len(skill.description) <= 1024


@pytest.mark.parametrize(
    ("frontmatter", "description"),
    [
        ("name: pdf\ndescription: Merge PDFs.", "Merge PDFs."),
        ("name: pdf\ndescription: 'Use when: merging PDFs'", "Use when: merging PDFs"),
        ("name: pdf\ndescription: Use when: merging PDFs", "Use when: merging PDFs"),
        ("name: pdf\ndescription: >-\n  Merge\n  PDFs.\nlicense: MIT", "Merge PDFs."),
        ('name: pdf\nmetadata:\n  author: me\ndescription: "Merge PDFs."', "Merge PDFs."),
    ],
)
def test_frontmatter_is_parsed_leniently(
    tmp_path: Path, frontmatter: str, description: str
) -> None:
    location = _write_skill(tmp_path, "pdf", frontmatter, body="---\nname: not-this\n---")
    assert skills._load(location) == skills.Skill("pdf", description, location.resolve())  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    "text", ["# No frontmatter", "---\nname: pdf\n---", "---\nname: pdf\ndescription:\n---"]
)
def test_skills_without_a_description_are_skipped(tmp_path: Path, text: str) -> None:
    location = tmp_path / "pdf" / "SKILL.md"
    location.parent.mkdir()
    location.write_text(text)
    assert skills._load(location) is None  # pyright: ignore[reportPrivateUsage]


def test_missing_name_falls_back_to_the_directory(tmp_path: Path) -> None:
    location = _write_skill(tmp_path, "pdf", "description: Merge PDFs.")
    assert skills._load(location) == skills.Skill("pdf", "Merge PDFs.", location.resolve())  # pyright: ignore[reportPrivateUsage]


def test_project_skills_override_bundled_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundled = tmp_path / "bundled"
    _write_skill(bundled, "a", "name: a\ndescription: Bundled a.")
    _write_skill(bundled, "b", "name: b\ndescription: Bundled b.")
    _write_skill(tmp_path / ".agents/skills", "b", "name: b\ndescription: Project b.")
    (tmp_path / ".agents/skills/notes.md").write_text("not a skill")
    monkeypatch.setattr(skills, "BUNDLED_SKILLS", bundled)

    found = {skill.name: skill.description for skill in skills.discover(tmp_path)}
    assert found == {"a": "Bundled a.", "b": "Project b."}


def test_catalog_is_empty_without_skills() -> None:
    assert skills.catalog([]) == ""


def test_catalog_lists_escaped_skill_metadata(tmp_path: Path) -> None:
    location = tmp_path / "x" / "SKILL.md"
    catalog = skills.catalog([skills.Skill("x", "Use for <html> & XML.", location)])
    assert catalog.startswith(skills.CATALOG_INSTRUCTIONS)
    assert "<description>Use for &lt;html&gt; &amp; XML.</description>" in catalog
    assert str(location) not in catalog


def test_activation_returns_body_directory_and_resources(tmp_path: Path) -> None:
    location = _write_skill(tmp_path, "pdf", "name: pdf\ndescription: PDFs.", body="# Merge")
    (location.parent / "scripts").mkdir()
    (location.parent / "scripts/merge.py").write_text("")
    content = skills.activate(skills.Skill("pdf", "PDFs.", location))

    assert content.startswith('<skill_content name="pdf">\n# Merge\n\n')
    assert "description:" not in content
    assert f"Skill directory: {location.parent}" in content
    assert "<file>scripts/merge.py</file>" in content and "<file>SKILL.md" not in content
    assert content.endswith("</skill_resources>\n</skill_content>")


def test_activation_without_resources_or_file(tmp_path: Path) -> None:
    location = _write_skill(tmp_path, "pdf", "name: pdf\ndescription: PDFs.")
    assert "<skill_resources>" not in skills.activate(skills.Skill("pdf", "PDFs.", location))
    location.unlink()
    assert "Could not read skill 'pdf'" in skills.activate(skills.Skill("pdf", "PDFs.", location))
