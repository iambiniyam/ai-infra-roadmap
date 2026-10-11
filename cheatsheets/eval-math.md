# Evaluation Math Cheatsheet

How many samples do you need, is that improvement real, and what does an agent's success rate actually mean? Evals are infrastructure: they have a cost, a confidence interval, and a failure mode.

---

## 1. The question that kills most eval claims

> "Is a 2-point win on a 500-item benchmark real?"

Usually: **no.** At 500 items the 95% confidence interval is roughly ±4.4 points, so a 2-point gap is entirely inside the noise. This section gives you the arithmetic to say that *before* you ship the claim.

---

## 2. Confidence intervals: use Wilson, not Wald

The textbook "Wald" interval (`p̂ ± z·√(p̂(1−p̂)/n)`) is **wrong for evals** because it collapses to zero width when a score is 0% or 100% and is badly biased at small *n*. Evals hit both cases constantly (a model that scores 0 on a task, a test suite that passes fully).

Use the **Wilson score interval**:

```text
        p̂ + z²/(2n)          z                        p̂(1−p̂)   z²
center = ───────────  ±  ─────────── · sqrt(  ───────  +  ──── )
         1 + z²/n          1 + z²/n            n        4n²
```

Wilson behaves correctly at the boundaries and is the right default. (For a quick gut check, `1/√n` approximates the half-width at p=0.5.)

**Worked, at p = 0.5, 95% confidence:**

| Items (n) | 95% CI half-width | Can it detect a 3-point gap? |
|---:|---:|---|
| 50 | ±13.4 pp | No |
| 100 | ±9.6 pp | No |
| 250 | ±6.2 pp | No |
| 500 | ±4.4 pp | No |
| 1,067 | ±3.0 pp | Barely |
| 5,000 | ±1.4 pp | Yes |

> **Rule of thumb:** to measure a rate to ±3 points you need about **1,000 items**. To ±1 point you need about **10,000**. Most public benchmarks are far smaller than the claims made from them.

---

## 3. Sizing an eval before you run it

Worst case (p = 0.5), single proportion:

```text
n ≈ z² · p(1−p) / e²        z = 1.96 for 95%

±5pp → ~385 items
±3pp → ~1,070 items
±2pp → ~2,400 items
±1pp → ~9,600 items
```

**Do this arithmetic before you build the eval.** It tells you whether your eval *can* answer the question you are asking. An eval of 100 items cannot detect a 3-point regression no matter how carefully you run it — you will ship the regression and see nothing.

### Comparing two systems

Comparing two models on the **same** items is a **paired** design, which is far more sensitive than comparing two independent proportions. Use **McNemar's test** on the discordant pairs (items where exactly one system succeeded), not a two-sample proportion test.

The sobering consequence: detecting small paired differences is *expensive*.

| Improvement | Items needed (roughly) |
|---|---|
| 10pp → 15pp | ~600 |
| 70% → 75% | ~630 |
| 70% → 73% | ~1,700 |
| 70% → 71% | ~15,000 |

**A 1-point difference needs far more items than most public benchmarks contain.** Any 1-2 point leaderboard win is, absent a paired design at scale, indistinguishable from noise. Say so when you see one.

---

## 4. `pass@k` vs `pass^k` — the agent metric that matters

These are different questions and only one of them describes production:

- **pass@k** — "does the agent succeed *at least once* in k tries?" Best-of-k. Rises with k. This is what most benchmark tables report.
- **pass^k** — "does the agent succeed *every* time across k independent tries?" Rises only if the agent is genuinely reliable. **Falls** as k grows.

For a *dependent* multi-step task you need **pass^k**. The collapse is brutal:

| k | pass^k at 70% per-trial success |
|---:|---:|
| 1 | 70.0% |
| 2 | 49.0% |
| 4 | 24.0% |
| 8 | **5.8%** |
| 16 | 0.3% |

A "70% agent" completes an 8-step dependent task about **6% of the time**. Any agent workflow with chained tool calls compounds its per-step reliability, and this is why agent products fail in ways their benchmarks do not predict.

```text
pass^k ≈ p^k            (p = per-trial success, k = dependent steps)
```

**Practical use:** quote pass^k for anything a user has to sit through end to end; quote pass@k only when a human or a filter selects the best attempt.

---

## 5. LLM-as-judge: measure the judge before you trust it

A judge is a model with biases. If you use one, quantify it:

- **Agreement with human labels**, using **Cohen's κ** (two raters, chance-corrected). κ ≈ 0.4-0.6 is moderate; do not treat a judge at κ = 0.3 as ground truth.
- For more than two raters or highly skewed label distributions, **Gwet's AC1** is more stable than Fleiss' κ — κ paradoxically collapses when agreement is very high and prevalence is skewed, which is exactly the regime of most eval sets.
- **Position bias** — swap the order of A and B, re-run, and report the flip rate. If the judge's preference follows position, the score is partly measuring ordering.
- **Verbosity bias** — longer answers score higher regardless of correctness. Test by padding a bad answer.
- **Self-preference** — a judge favors outputs from its own model family. Never use the model under test as its own judge.

**Report the judge's reliability next to the score it produced.** A number from an unvalidated judge is not evidence.

---

## 6. Eval-set hygiene

- **Contamination:** check whether your eval items appear in training data (n-gram overlap, or ask the model to complete the item from its prefix). Contaminated evals measure memorization.
- **Variance/flakiness:** run the same eval twice on the same model. The gap between runs is **your noise floor** — no improvement smaller than that is real. Temperature > 0 makes this larger.
- **Freeze the set, version it, and log the hash** alongside results. A changed eval set silently invalidates every prior number.
- **Hold out a private set** you never tune against, or you are doing gradient descent on your benchmark.

---

## 7. Cost math for evals

```text
cost = items × (avg_input_tokens × in_price + avg_output_tokens × out_price)
       × number of configs swept
```

Two things that surprise people:

1. Sweeping configs multiplies cost linearly — a 5-config sweep on a 1,000-item eval is 5,000 generations, not 1,000.
2. **Agentic evals are multiplied again by turns.** An agent that averages 40 turns with growing context re-sends history every turn, so input tokens grow roughly **quadratically** in turn count.

```text
total_input_tokens ≈ n·L₀ + a·n(n−1)/2
  n = turns, L₀ = per-turn new content, a = growth per turn
```

At a large system prompt plus tool results over dozens of turns, the overwhelming majority of input tokens are **re-sent history**. This is why prefix caching is the single highest-leverage optimization for agent evals — and why anything that breaks the cache (including aggressive context compaction every turn) can cost more than it saves.

---

## 8. Checklist before publishing an eval number

- [ ] n is large enough for the effect you claim (Section 3), or you claim nothing.
- [ ] CI reported using Wilson, not Wald.
- [ ] Comparisons are paired (McNemar), not two independent proportions.
- [ ] Noise floor measured by re-running the same config.
- [ ] Judge (if any) validated against humans, with κ/AC1 and position-bias flip rate reported.
- [ ] Eval set hash, version, and contamination check recorded.
- [ ] Agent results quoted as pass^k for the end-to-end path.
- [ ] Cost and total generation count stated.

---

**Back to:** [README](../README.md) · [ROADMAP](../ROADMAP.md) · [inference math](inference-math.md) · [RL post-training](rl-post-training.md)
