#!/usr/bin/env python3
"""Refresh GitHub star counts in this repo's Markdown tables.

Finds rows shaped like:

    | [Name](https://github.com/owner/repo) | 12,345 | Apache-2.0 | notes |
    | [Name](https://github.com/owner/repo) | 103.7k | notes |

and rewrites only the star cell. All other formatting (and the license/notes
columns) is preserved verbatim, so the diff stays reviewable.

Usage:
    GITHUB_TOKEN=ghp_... python3 scripts/refresh_stars.py [--check]

`--check` exits non-zero if any count is stale (useful in CI without push perms).
Without a token it falls back to the unauthenticated API (60 requests/hour),
which is usually enough because repos are deduplicated and batched.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPHQL = "https://api.github.com/graphql"
BATCH = 80

# | [text](https://github.com/owner/repo) | <stars> |
ROW_RE = re.compile(
    r"^\|\s*\[(?P<label>[^\]]+)\]\(https://github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)\)\s*\|"
    r"\s*(?P<stars>~?[\d.,]+k?)\s*\|"
)

# Lines that advertise when the data was last refreshed. Keeping these in sync
# with the actual run stops the docs from quietly going stale.
DATE_PATTERNS = [
    re.compile(r"(\*\*Last full data refresh:\*\* )(\d{4}-\d{2}-\d{2})"),
    re.compile(r"(pulled from the GitHub API on \*\*)(\d{4}-\d{2}-\d{2})(\*\*)"),
]


def md_files() -> list[pathlib.Path]:
    files = [ROOT / "STACK.md", ROOT / "README.md", ROOT / "RESOURCES.md"]
    files += sorted(ROOT.glob("cheatsheets/*.md"))
    return [f for f in files if f.exists()]


def collect(files: list[pathlib.Path]) -> dict[str, str]:
    """Return {full_name: original_star_cell} for every table row found."""
    found: dict[str, str] = {}
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            m = ROW_RE.match(line)
            if m:
                found[f"{m['owner']}/{m['repo']}".lower()] = m["stars"]
    return found


def format_stars(stars: int, original: str) -> str:
    """Match the existing style: thousands separators, or compact k-notation."""
    if "k" in original.lower():
        prefix = "~" if original.strip().startswith("~") else ""
        return f"{prefix}{stars / 1000:.1f}k"
    return f"{stars:,}"


def fetch_counts(repos: list[str], token: str | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    headers = {"Accept": "application/json", "User-Agent": "ai-infra-roadmap-bot"}
    if token:
        headers["Authorization"] = f"bearer {token}"

    for start in range(0, len(repos), BATCH):
        chunk = repos[start : start + BATCH]
        parts = []
        for i, full in enumerate(chunk):
            owner, _, name = full.partition("/")
            parts.append(
                f'r{i}: repository(owner: "{owner}", name: "{name}") '
                "{ stargazerCount }"
            )
        body = json.dumps({"query": "{ " + " ".join(parts) + " }"}).encode()
        req = urllib.request.Request(GRAPHQL, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.load(resp)
        except urllib.error.HTTPError as exc:
            sys.exit(f"GitHub API error {exc.code}: {exc.read().decode()[:400]}")
        except urllib.error.URLError as exc:
            sys.exit(f"Network error talking to GitHub: {exc}")

        if "errors" in payload:
            # Individual missing repos come back as per-alias NOT_FOUND; ignore those.
            hard = [e for e in payload["errors"] if e.get("type") != "NOT_FOUND"]
            if hard:
                sys.exit(f"GraphQL error: {hard[:2]}")

        for i, full in enumerate(chunk):
            node = (payload.get("data") or {}).get(f"r{i}")
            if node:
                counts[full] = int(node["stargazerCount"])
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if anything is stale")
    args = ap.parse_args()

    files = md_files()
    current = collect(files)
    if not current:
        sys.exit("No star rows found — did the table format change?")

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    counts = fetch_counts(sorted(current), token)
    print(f"Fetched {len(counts)}/{len(current)} repositories.")

    stale: list[str] = []
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        changed = False
        for idx, line in enumerate(lines):
            for pattern in DATE_PATTERNS:
                if pattern.search(line):
                    new_line = pattern.sub(
                        lambda m: m.group(1) + today + (m.group(3) if m.lastindex == 3 else ""),
                        line,
                    )
                    if new_line != line:
                        lines[idx] = line = new_line
                        changed = True
            m = ROW_RE.match(line)
            if not m:
                continue
            full = f"{m['owner']}/{m['repo']}".lower()
            if full not in counts:
                continue
            new = format_stars(counts[full], m["stars"])
            if new != m["stars"]:
                stale.append(f"{full}: {m['stars']} -> {new}")
                lines[idx] = line.replace(
                    f"| {m['stars']} |", f"| {new} |", 1
                )
                changed = True
        if changed and not args.check:
            path.write_text("".join(lines), encoding="utf-8")
            print(f"  updated {path.relative_to(ROOT)}")

    if stale:
        print(f"\n{len(stale)} stale star count(s):")
        for item in stale[:40]:
            print(f"  {item}")
        if args.check:
            return 1
    else:
        print("All star counts are current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
