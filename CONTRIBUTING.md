# Contributing

Thanks for helping make this the best free AI-infra roadmap. The bar is **clarity and correctness**, not volume.

## What we want

1. **A genuinely better path.** If you can reorder a stage, add an exit criterion, or replace a fuzzy instruction with a measurable one — that's the highest-value contribution.
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
```

Then:

- Keep the diff focused. One logical change per PR.
- Don't renumber existing projects or reorder stages in a drive-by PR — open an issue first.
- Match the existing voice: direct, concrete, and allergic to hype.

## Style guide

- **Concrete over abstract.** Prices, tokens/sec, and percentiles beat adjectives.
- **State the tradeoff.** Every recommendation should say when it *doesn't* apply.
- **No marketing language.** "Blazing fast" is noise; "2.3x lower p99 TTFT in our benchmark" is signal.
- **Mark uncertainty.** If something is version-dependent or rumored, say so.
- **Keep lines readable.** Tables over walls of text.

## Reporting problems

Bug (broken link, wrong number, bad command) → open an issue with the file and line.
Disagreement with a recommendation → open an issue with evidence. We'd rather be corrected than consistent.

## License

By contributing you agree your contributions are licensed under the MIT License (see [LICENSE](LICENSE)).
