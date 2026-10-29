"""Docs stay navigable: relative links resolve, every journal entry is indexed and has a status."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "docs" / "journal"
LINK = re.compile(r"\]\(([^)#\s]+)")


def test_relative_links_resolve() -> None:
    broken = [
        f"{md.relative_to(ROOT)} -> {target}"
        for md in ROOT.glob("**/*.md")
        if ".venv" not in md.parts
        for target in LINK.findall(md.read_text())
        if "://" not in target and not (md.parent / target).exists()
    ]
    assert not broken


def test_journal_entries_are_indexed_with_status() -> None:
    index = (JOURNAL / "README.md").read_text()
    for entry in JOURNAL.glob("2*.md"):
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-[a-z0-9-]+\.md", entry.name)
        assert f"({entry.name})" in index, f"{entry.name} missing from journal index"
        assert "**Status:**" in entry.read_text(), f"{entry.name} has no status"
