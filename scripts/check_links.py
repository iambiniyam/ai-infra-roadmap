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
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
URL_RE = re.compile(r"https?://[^\s\)\]\"<>`]+")

# Network errors that mean "we could not reach it", not "it is gone". These get
# retried and, if they persist, reported as warnings instead of failures --
# otherwise one transient TLS reset reddens CI on a perfectly live link.
TRANSIENT = (
    "URLError",
    "TimeoutError",
    "ConnectionResetError",
    "RemoteDisconnected",
    "IncompleteRead",
    "SSLError",
    "ssl.SSLError",
    "socket.timeout",
    "http.client.RemoteDisconnected",
)

RETRIES = 3

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


def probe_once(url: str) -> tuple[str, int | str]:
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


def curl_probe(url: str) -> int | str:
    """Fall back to curl, whose TLS stack handles more hosts than Python's."""
    if not shutil.which("curl"):
        return "unreachable"
    try:
        proc = subprocess.run(
            ["curl", "-sSL", "-o", "/dev/null", "-w", "%{http_code}",
             "--max-time", "30", "-A", UA, url],
            capture_output=True,
            text=True,
            timeout=45,
        )
        code = proc.stdout.strip()
        return int(code) if code.isdigit() and code != "000" else "unreachable"
    except Exception:  # noqa: BLE001
        return "unreachable"


def probe(url: str) -> tuple[str, int | str, bool]:
    """Return (url, status, is_transient).

    Retries transient failures with backoff, then tries curl, so a live link is
    not reported as broken because of one flaky handshake.
    """
    status: int | str = "unreachable"
    for attempt in range(RETRIES):
        _, status = probe_once(url)
        if isinstance(status, int):
            return url, status, False
        if status not in TRANSIENT and status != "unreachable":
            return url, status, False
        time.sleep(1.5 * (attempt + 1))
    code = curl_probe(url)
    if isinstance(code, int):
        return url, code, False
    return url, status, True


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
        for url, status, transient in pool.map(probe, targets):
            ok = isinstance(status, int) and 200 <= status < 400
            if ok:
                continue
            if isinstance(status, int) and status in allowed:
                warnings.append((url, status, links[url]))
            elif transient:
                # Could not establish anything after retries and a curl fallback.
                warnings.append((url, "unreachable", links[url]))
            elif status == "unreachable" and any(h in url for h in SKIP_HOSTS):
                warnings.append((url, status, links[url]))
            else:
                failures.append((url, status, links[url]))

    if warnings:
        print(f"\n⚠️  {len(warnings)} warning(s) (bot-blocked or unreachable, not necessarily dead):")
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
