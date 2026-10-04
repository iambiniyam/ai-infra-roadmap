#!/usr/bin/env python3
"""Verify that every relative link and '#anchor' in the docs resolves.

GitHub generates heading anchors by lowercasing, stripping punctuation, and
replacing spaces with hyphens. That algorithm is easy to get subtly wrong
(em-dashes leave a double hyphen), so this checks it mechanically.

Usage:
    python3 scripts/check_internal_links.py
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)
LINK_RE = re.compile(r"\[[^\]]*\]\((?P<target>[^)\s]+)\)")


def slug(heading: str) -> str:
    """Reproduce GitHub's heading-anchor slug algorithm (github-slugger).

    Order matters: lowercase, trim, strip punctuation, then replace each ASCII
    space with a hyphen. An em-dash surrounded by spaces therefore leaves TWO
    hyphens ('Stage 0 — Foundations' -> 'stage-0--foundations'), and a leading
    emoji leaves a leading hyphen.
    """
    text = heading.strip()
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)      # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links -> label
    text = re.sub(r"`([^`]*)`", r"\1", text)              # code ticks
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)                  # strip punctuation
    return text.replace(" ", "-")


def anchors(path: pathlib.Path) -> set[str]:
    """All anchors GitHub generates, including -1/-2 suffixes for duplicates."""
    body = path.read_text(encoding="utf-8")
    seen: dict[str, int] = {}
    out: set[str] = set()
    for match in HEADING_RE.finditer(body):
        base = slug(match.group(2))
        count = seen.get(base, 0)
        seen[base] = count + 1
        out.add(base if count == 0 else f"{base}-{count}")
    return out


def main() -> int:
    files = sorted(ROOT.rglob("*.md"))
    anchor_map = {f: anchors(f) for f in files}
    problems: list[str] = []
    checked = 0

    for path in files:
        text = path.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            target = match.group("target")
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            checked += 1
            file_part, _, frag = target.partition("#")
            dest = (path.parent / file_part).resolve() if file_part else path

            if file_part and not dest.exists():
                problems.append(f"{path.relative_to(ROOT)} -> missing file '{file_part}'")
                continue
            if frag:
                known = anchor_map.get(dest)
                if known is None:
                    problems.append(
                        f"{path.relative_to(ROOT)} -> cannot read '{file_part}'"
                    )
                elif frag.lower() not in known:
                    problems.append(
                        f"{path.relative_to(ROOT)} -> dead anchor '#{frag}' in {file_part or path.name}"
                    )

    print(f"Checked {checked} internal links across {len(files)} files.")
    if problems:
        print(f"\n❌ {len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        return 1
    print("\n✅ All internal links and anchors resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
