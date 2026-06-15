from __future__ import annotations

from pathlib import Path

class PromptCatalog:
    def __init__(self, prompt_root: Path):
        self.prompt_root = prompt_root

    def get(self, name: str) -> str:
        return _load_prompt(self.prompt_root, name)


def _load_prompt(prompt_root: Path, name: str) -> str:
    path = prompt_root / f"{name}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()
