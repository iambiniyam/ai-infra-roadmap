# Contributing

Thanks for helping make this the best free AI-infra roadmap. The bar is **clarity and correctness**, not volume.

## What we want

1. **A genuinely better path.** If you can reorder a stage, add an exit criterion, or replace a fuzzy instruction with a measurable one, that's the highest-value contribution.
2. **Missing free resources.** Courses, textbooks, papers, lectures. Must be free to access (or clearly marked *paid*).
3. **Corrections.** A wrong formula, a stale claim, a link that rots, a license mislabeled. These matter a lot.
4. **Projects with acceptance criteria.** "Build X" is not a project. "Build X and be able to show Y, measured by Z" is.

## What we'll likely decline

- Tools added purely because they're popular (the repo has a *stack*, not a directory).
- Paid-only resources without an explicit *(paid)* label.
- Anything that duplicates an existing row without a clear "use this when" distinction.
- Content generated without being tested against a real system.

## Adding a project to `STACK.md`

Find the right layer table and add a row in this exact shape:

```markdown
| [owner/repo](https://github.com/owner/repo) | 12,345 | Apache-2.0 | One sentence: what it does and **when to pick it**. |
```

Rules:

- Star count goes in as a plain number (the [refresh bot](.github/workflows/refresh-stars.yml) corrects it weekly).
- License must be the SPDX id from the GitHub API, or `NOASSERTION` if GitHub can't detect it.
- The note must say *when to use it*, not just what it is.
- Prefer projects that are maintained (a commit in the last ~9 months) and have docs.
- If a project is source-available rather than OSI-open-source (BSL/SSPL/fair-code), say so in the note.

## Adding a resource to `RESOURCES.md`

Include: name, a working link, who it's for, and one sentence on why it's worth the hours. If it's paid, mark it *(paid)*.

## Adding a project to `PROJECTS.md`

Use this structure:

```markdown
### P<n> · <Name>

**Time:** <duration> · **Skills:** Stage <n>

<One-sentence problem statement.>

Deliverables:
- ...

**Acceptance criteria**
- [ ] A criterion you can verify without judgment.

**Proves:** <the skill this demonstrates>
```

## Before you open a PR

```bash
# 1. Validate links (GitHub-only is fast and reliable)
python3 scripts/check_links.py --only-github

# 2. Refresh star counts
GITHUB_TOKEN=$(gh auth token) python3 scripts/refresh_stars.py

# 3. Check archive/license/maintenance claims against the live API
GITHUB_TOKEN=$(gh auth token) python3 scripts/check_health.py
```

CI runs all of these plus the internal anchor checker on every PR.

### What the health checker enforces

`check_health.py` implements the rules above automatically, because a curated stack rots silently:

- **Archived projects must say so in their own note.** Archiving is not an automatic removal — several entries here are archived-but-canonical (`AutoGPTQ`, `TGI`) or finished works (`llm.c`). The rule is **disclosure, not deletion**: if it is archived, the note must admit it and point at a maintained alternative.
- **Quiet projects** (no push in ~9 months) should be noted or replaced. Stable-and-complete teaching repos are fine; note them as such.
- **License cells must match the SPDX id** the GitHub API reports.
- **Links must resolve** — no 404s or stale slugs after a rename.

A stale GitHub slug is the fastest way to lose a reader's trust, and org renames are common in this ecosystem (for example, `verl` moved to the `verl-project` org in 2026). Always use the canonical owner.

Then:

- Keep the diff focused. One logical change per PR.
- Don't renumber existing projects or reorder stages in a drive-by PR; open an issue first.
- Match the existing voice: direct, concrete, and allergic to hype.
- **Cite primary sources for factual claims.** If you assert a hardware spec, a benchmark number, or a paper result, link the datasheet, the paper, or the upstream repo. Verify arXiv IDs by fetching them — do not cite from memory.

## Style guide

- **Concrete over abstract.** Prices, tokens/sec, and percentiles beat adjectives.
- **State the tradeoff.** Every recommendation should say when it *doesn't* apply.
- **No marketing language.** "Blazing fast" is noise; "2.3x lower p99 TTFT in our benchmark" is signal.
- **Mark uncertainty.** If something is version-dependent or rumored, say so.
- **Separate durable from volatile.** Formulas and ratios last; product names and prices rot. Date-stamp the volatile ones.
- **Keep lines readable.** Tables over walls of text.

## Reporting problems

Bug (broken link, wrong number, bad command) → open an issue with the file and line.
Disagreement with a recommendation → open an issue with evidence. We'd rather be corrected than consistent.

## License

By contributing you agree your contributions are licensed under the MIT License (see [LICENSE](LICENSE)).
