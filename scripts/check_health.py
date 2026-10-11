#!/usr/bin/env python3
"""Enforce the maintenance bar that CONTRIBUTING.md promises.

CONTRIBUTING.md says: "Prefer projects that are maintained (a commit in the
last ~9 months)" and "If a project is source-available rather than
OSI-open-source (BSL/SSPL/fair-code), say so in the note." Nothing checked
either claim. This does, against the live GitHub API.

For every `| [name](https://github.com/owner/repo) | ... |` row in STACK.md and
README.md it reports:

  * archived repos that do not say so in their own note
  * repos with no push in N months that do not say so in their own note
  * license cells that disagree with the SPDX id GitHub reports
  * rows whose links 404 or have been renamed

Design note: an archived or quiet repo is *not* automatically a failure. Several
of the best entries here are archived-but-canonical (AutoGPTQ, TGI) or finished
works (llm.c). The rule is disclosure, not deletion: if it is archived, the note
must admit it. That keeps curation honest without churning the stack.

Usage:
    GITHUB_TOKEN=$(gh auth token) python3 scripts/check_health.py
    GITHUB_TOKEN=$(gh auth token) python3 scripts/check_health.py --stale-months 9

Exit code is non-zero only on undisclosed archived/renamed rows and license
mismatches. Stale-but-disclosed is a warning.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPHQL = "https://api.github.com/graphql"
BATCH = 50
DEFAULT_FILES = ("STACK.md", "README.md")

# Any table row whose first cell is a github.com link. Captures the whole line
# so the note column can be inspected for an existing disclosure.
ROW_RE = re.compile(
    r"^\|\s*\[(?P<label>[^\]]+)\]\(https://github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)(?:[^)]*)\)"
    r"(?P<rest>.*)$"
)

# Phrases that count as an honest disclosure in the note column.
DISCLOSURE = re.compile(
    r"archived|unmaintained|no longer maintained|superseded|deprecated|"
    r"finished|\bdormant\b|stable and complete|read(?:-| )only|"
    r"successor|reference only|historically",
    re.IGNORECASE,
)

# Column layout differs between files, so pull license/note out positionally.
STACK_COLS = re.compile(r"^\s*\|\s*(?P<stars>~?[\d.,]+k?)\s*\|\s*(?P<license>[^|]*?)\s*\|\s*(?P<note>.*?)\s*\|\s*$")


def github_token() -> str | None:
    t = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if t:
        return t
    try:
        return subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def gql(tok: str, query: str, variables: dict) -> dict:
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(
        GRAPHQL,
        data=body,
        headers={
            "Authorization": f"bearer {tok}",
            "Content-Type": "application/json",
            "User-Agent": "ai-infra-roadmap-health-check",
        },
    )
    last: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 502, 503) and attempt < 3:
                time.sleep(2**attempt)
                continue
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < 3:
                time.sleep(2**attempt)
                continue
            raise
    raise RuntimeError(f"GraphQL failed: {last}")


def parse_rows(paths: list[pathlib.Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = ROW_RE.match(line)
            if not m:
                continue
            rest = m.group("rest").lstrip("|").strip()
            cols = STACK_COLS.match(rest) if path.name == "STACK.md" else None
            rows.append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "line": i,
                    "label": m.group("label"),
                    "slug": f"{m.group('owner')}/{m.group('repo')}",
                    "license_cell": (cols.group("license") if cols else "").strip(),
                    "note": (cols.group("note") if cols else rest),
                    "line_text": line,
                }
            )
    return rows


def fetch(tok: str, slugs: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    uniq = sorted({s.lower() for s in slugs})
    for i in range(0, len(uniq), BATCH):
        chunk = uniq[i : i + BATCH]
        parts, variables = [], {}
        for j, slug in enumerate(chunk):
            owner, name = slug.split("/", 1)
            parts.append(
                f"r{j}: repository(owner: $o{j}, name: $n{j}) "
                "{ nameWithOwner stargazerCount isArchived pushedAt licenseInfo { spdxId } }"
            )
            variables[f"o{j}"] = owner
            variables[f"n{j}"] = name
        decl = ", ".join(f"$o{j}: String!, $n{j}: String!" for j in range(len(chunk)))
        payload = gql(tok, f"query({decl}) {{ {' '.join(parts)} }}", variables)
        for j, slug in enumerate(chunk):
            out[slug] = payload.get("data", {}).get(f"r{j}") or {"missing": True}
    return out


def spdx_of(meta: dict) -> str:
    return ((meta.get("licenseInfo") or {}).get("spdxId") or "NOASSERTION")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale-months", type=int, default=9,
                    help="CONTRIBUTING.md's stated bar is ~9 months")
    ap.add_argument("--files", nargs="*", default=list(DEFAULT_FILES))
    args = ap.parse_args()

    tok = github_token()
    if not tok:
        print("No GitHub token (set GITHUB_TOKEN or log in with gh).", file=sys.stderr)
        return 2

    paths = [ROOT / f for f in args.files]
    rows = parse_rows(paths)
    meta = fetch(tok, [r["slug"] for r in rows])

    now = dt.datetime.now(dt.timezone.utc)
    undisclosed_archived, undisclosed_stale = [], []
    license_mismatch, missing = [], []
    disclosed_archived = 0

    for r in rows:
        m = meta.get(r["slug"].lower(), {})
        if not m.get("nameWithOwner"):
            missing.append(r)
            continue

        has_disclosure = bool(DISCLOSURE.search(r["note"] or "")) or bool(
            DISCLOSURE.search(r["line_text"])
        )
        pushed = m.get("pushedAt")
        months = None
        if pushed:
            months = (now - dt.datetime.fromisoformat(
                pushed.replace("Z", "+00:00"))).days / 30.44

        if m.get("isArchived"):
            if has_disclosure:
                disclosed_archived += 1
            else:
                undisclosed_archived.append((r, m))
        elif months is not None and months > args.stale_months and not has_disclosure:
            undisclosed_stale.append((r, m, months))

        if r["license_cell"] and r["license_cell"].lower() != spdx_of(m).lower():
            license_mismatch.append((r, r["license_cell"], spdx_of(m)))

    if undisclosed_archived:
        print(f"❌ {len(undisclosed_archived)} archived repo(s) not disclosed in their note:")
        for r, m in undisclosed_archived:
            print(f"   {r['file']}:{r['line']}  {r['slug']}  "
                  f"(last push {m['pushedAt'][:10]}, {m['stargazerCount']:,} stars)")
        print("   -> add 'Archived.' plus a maintained alternative to the note, "
              "or drop the row.\n")

    if license_mismatch:
        print(f"❌ {len(license_mismatch)} license cell(s) disagree with the GitHub API:")
        for r, cell, api in license_mismatch:
            print(f"   {r['file']}:{r['line']}  {r['slug']}  cell={cell!r} api={api!r}\n")

    if missing:
        print(f"❌ {len(missing)} row(s) point at repos that 404 or were renamed:")
        for r in missing:
            print(f"   {r['file']}:{r['line']}  {r['slug']}\n")

    if undisclosed_stale:
        print(f"⚠️  {len(undisclosed_stale)} undisclosed quiet repo(s) "
              f"(no push in >{args.stale_months} months):")
        for r, m, months in undisclosed_stale:
            print(f"   {r['file']}:{r['line']}  {r['slug']}  "
                  f"last push {m['pushedAt'][:10]} ({months:.0f} mo)")
        print("   -> CONTRIBUTING.md asks for a commit in the last ~9 months. "
              "Note it, or replace it.\n")

    print(f"Checked {len(rows)} rows / {len(meta)} repos · "
          f"{disclosed_archived} archived-but-disclosed (fine) · "
          f"{len(license_mismatch)} license mismatches · "
          f"{len(undisclosed_archived)} undisclosed archives")

    return 1 if (undisclosed_archived or license_mismatch or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
