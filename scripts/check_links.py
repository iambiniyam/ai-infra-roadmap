#!/usr/bin/env python3
"""Check every link in the Markdown for rot.

Usage:
    python3 scripts/check_links.py                # check everything
    python3 scripts/check_links.py --only-github  # fastest, most reliable
    python3 scripts/check_links.py --allow 403,405

Exit code is non-zero when a link returns a hard failure (404/410/5xx).
Sites that are known to block bots (403/405/429) are reported as warnings so
CI stays useful instead of crying wolf.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
URL_RE = re.compile(r"https?://[^\s\)\]\"<>`]+")

# Hosts that aggressively block automated requests; we can't tell rot from policy.
SKIP_HOSTS = (
    "twitter.com",
    "x.com",
    "discord.gg",
    "reddit.com",
    "www.reddit.com",
    "youtube.com",
    "www.youtube.com",
    "linkedin.com",
    "openreview.net",
    "dl.acm.org",
    "www.oreilly.com",
    "ocw.mit.edu",
    "mlsys.org",
    "mlcommons.org",
)

# Documented placeholders used as examples, not real links.
PLACEHOLDER_URLS = {"https://github.com/owner/repo"}

UA = "Mozilla/5.0 (compatible; ai-infra-roadmap-link-checker/1.0)"


def markdown_files() -> list[pathlib.Path]:
    return sorted(ROOT.rglob("*.md"))


def urls(files: list[pathlib.Path], only_github: bool) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in files:
        text = path.read_text(encoding="utf-8")
        for url in URL_RE.findall(text):
            url = url.rstrip(".,;:")
            if only_github and "github.com" not in url:
                continue
            found.setdefault(url, set()).add(str(path.relative_to(ROOT)))
    return found


def probe(url: str) -> tuple[str, int | str]:
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, method=method, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                return url, resp.status
        except urllib.error.HTTPError as exc:
            if method == "HEAD" and exc.code in (403, 405, 501):
                continue  # many servers reject HEAD; retry with GET
            return url, exc.code
        except Exception as exc:  # noqa: BLE001 - network is messy by nature
            if method == "HEAD":
                continue
            return url, type(exc).__name__
    return url, "unreachable"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-github", action="store_true")
    ap.add_argument("--allow", default="403,405,429",
                    help="comma-separated HTTP codes to treat as warnings")
    args = ap.parse_args()

    allowed = {int(c) for c in args.allow.split(",") if c.strip().isdigit()}
    links = urls(markdown_files(), args.only_github)
    links = {u: w for u, w in links.items() if u not in PLACEHOLDER_URLS}
    targets = [u for u in links if not any(h in u for h in SKIP_HOSTS)]
    print(f"Found {len(links)} unique links; checking {len(targets)} "
          f"(skipping {len(links) - len(targets)} bot-hostile hosts).")

    failures: list[tuple[str, object, set[str]]] = []
    warnings: list[tuple[str, object, set[str]]] = []
    with futures.ThreadPoolExecutor(max_workers=16) as pool:
        for url, status in pool.map(probe, targets):
            ok = isinstance(status, int) and 200 <= status < 400
            if ok:
                continue
            if isinstance(status, int) and status in allowed:
                warnings.append((url, status, links[url]))
            elif status == "unreachable" and any(h in url for h in SKIP_HOSTS):
                warnings.append((url, status, links[url]))
            else:
                failures.append((url, status, links[url]))

    if warnings:
        print(f"\n⚠️  {len(warnings)} warning(s) (bot-blocked, not necessarily dead):")
        for url, status, where in warnings[:25]:
            print(f"  [{status}] {url}  ({', '.join(sorted(where))})")

    if failures:
        print(f"\n❌ {len(failures)} broken link(s):")
        for url, status, where in failures:
            print(f"  [{status}] {url}  ({', '.join(sorted(where))})")
        return 1

    print("\n✅ No broken links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
