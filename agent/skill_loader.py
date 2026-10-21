"""Discover metadata first; load complete skill instructions only when selected."""
import re
from pathlib import Path
import yaml
from config import settings

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
FRONT_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)


def skills_root() -> Path:
    return settings.resolve_path(settings.skills_path).resolve()


def read_metadata(path: Path) -> dict:
    data = path.read_text(encoding="utf-8")
    match = FRONT_RE.match(data)
    if not match:
        raise ValueError(f"Missing YAML frontmatter: {path}")
    meta = yaml.safe_load(match.group(1))
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid metadata: {path}")
    name, desc = meta.get("name"), meta.get("description")
    if not isinstance(name, str) or not NAME_RE.fullmatch(name) or name != path.parent.name:
        raise ValueError(f"Invalid skill name or directory mismatch: {path}")
    if not isinstance(desc, str) or not desc.strip():
        raise ValueError(f"Missing description: {path}")
    return {"name": name, "description": desc.strip()}


def discover_skills() -> list[dict]:
    root = skills_root()
    if not root.is_dir():
        raise FileNotFoundError(f"Skill directory not found: {root}")
    return [read_metadata(p) for p in sorted(root.glob("*/SKILL.md"))]


def load_skill(name: str) -> str:
    """Allow only discovered skill names; reject arbitrary filesystem paths."""
    if not NAME_RE.fullmatch(name):
        raise ValueError("Invalid skill name")
    root = skills_root()
    path = (root / name / "SKILL.md").resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"Skill not found: {name}")
    read_metadata(path)
    return path.read_text(encoding="utf-8")
