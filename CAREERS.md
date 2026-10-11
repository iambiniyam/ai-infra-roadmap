# Careers

How the skills in this repo turn into a job: the roles that exist, what interviews actually test, which artifacts get you taken seriously, and how to break in from where you already are.

> **Honesty note:** everything here about hiring is *pattern-based*, not audited data. Compensation in particular changes fast and varies wildly by region and company stage. Where a number is a rough range rather than a verified figure, it is labelled. Treat structure as durable and numbers as perishable.

---

## 1. The roles (and what they actually do)

"AI infra" is not one job. The day-to-day differs enough that picking the wrong one wastes years.

| Role | The actual job | Day-to-day | Bottleneck skill |
|---|---|---|---|
| **Inference / serving engineer** | Make a model answer faster and cheaper at a quality bar | Benchmarking, tuning engines, quantization, caching, latency tails, capacity | Latency/throughput/cost tradeoffs |
| **Performance / kernel engineer** | Make specific operations faster on specific silicon | Profiling, CUDA/Triton kernels, fusion, numerics, SASS-level reasoning | Deep hardware + measurement |
| **Distributed training engineer** | Make big training runs converge fast without falling over | Parallelism strategy, collectives, checkpointing, fault tolerance, MFU | Collective communication + failure modes |
| **ML platform engineer** | Run the shared GPU platform other teams build on | Scheduling, quotas, multi-tenancy, storage, CI/CD for models, observability | Systems design + other teams' trust |
| **GPU SRE / infrastructure ops** | Keep the fleet healthy and the incidents short | Health checks, burn-in, triage, capacity, on-call, vendor RMAs | Diagnosis under pressure |
| **Data / training-data engineer** | Feed the GPUs; build the data plane | Ingestion, dedup, tokenization, sharding, lineage, throughput | Pipelines at scale |
| **ML systems researcher** | Move the state of the art in systems | Reading/reproducing papers, novel scheduling or kernels, publishing | Research taste + rigor |
| **AI infra PM / TPM** | Decide what gets built and unblock it | Roadmaps, vendor strategy, cross-team dependencies | Judgment + communication |

**Which are growing:** inference/serving, platform, and performance work — the ratio of "train a model once" to "serve it a billion times" keeps shifting toward serving. **Which are narrowest:** pure benchmark-tuning roles that a framework release can obsolete.

**The honest advice:** pick the layer where you want depth and stay adjacent to one more. The most employable shape is *one layer deep, neighbors shallow* — see [the map](README.md#the-map).

---

## 2. What interviews actually test

Across most companies, AI-infra loops draw from five question families. Very little of it is trivia; almost all of it is *reasoning out loud under constraints*.

### Family 1 — Back-of-envelope / "estimate this"

The single most common screen, because it reveals whether you think in bytes and time.

- How much VRAM does a 70B model need to serve at 32k context, batch 32? Walk me through it.
- How many GPUs to train a 13B model on 2T tokens in 3 weeks?
- At what batch size does this workload become compute-bound instead of bandwidth-bound?
- Estimate the $/1M tokens for serving this model on this hardware.

**Strong answers** state assumptions explicitly, use the formulas in [HARDWARE.md](HARDWARE.md), get an order of magnitude, then **sanit-check it** ("a 70B in bf16 is 140 GB, so it cannot fit on one 80 GB card — that already decides the topology"). Getting a factor of 2 wrong is fine; failing to notice an impossible answer is not.

### Family 2 — Systems design

- Design a multi-tenant inference platform for 20 teams with different SLOs.
- Design the serving stack for a chat product with p95 TTFT < 500 ms.
- Design a training pipeline that survives a node failure every 3 hours.
- How would you serve 100 fine-tuned adapters cheaply?
- Design an eval pipeline that gates deploys.

**Strong answers** start by asking about the *workload* (QPS, prompt/output length distribution, SLO, burstiness, budget), then choose a design and name the tradeoff they just accepted, then name what they would measure to know if they were wrong. Weak answers start by naming tools.

### Family 3 — Debugging / "it's broken"

- Training hangs across nodes. Walk me through your diagnosis.
- Throughput dropped 30% overnight with no code change. What do you check?
- Loss spikes at step 40k. Diagnose.
- The serving endpoint's p99 is 10x p50. Why?

**Strong answers** are *bisecting* and *layered*: reproduce, narrow the layer, test one hypothesis at a time, and know the specific artifacts (`nccl-tests`, flight recorder, `nvidia-smi -q`, per-rank step times). They also distinguish fail-stop from **fail-slow**, which is the failure mode most candidates miss.

### Family 4 — Coding and implementation

- Implement attention (or a fused/simple kernel) with correct masking.
- Write a KV-cache allocator with paging.
- Given a slow data loader, profile and fix it.
- Implement an all-reduce-free ring or a simple collective.

**Strong answers** are correct first, then fast, and they narrate the memory hierarchy as they go. For kernel roles, expect a Triton or CUDA exercise with a real roofline argument.

### Family 5 — Judgment / tradeoffs

- When would you *not* use Kubernetes for this?
- When is disaggregated serving a bad idea?
- Your team wants to fine-tune; when is prompting or RAG the better call?
- You have budget for either 2x more GPUs or 1 more engineer. Which?
- A model update regresses quality but improves latency. Ship it?

**Strong answers** are conditional and cost-aware, and they name the measurement that would resolve the question. There is rarely a single right answer; there is a clearly *reasoned* one.

### How to practice

Do the interview questions as **labs**, not flashcards. Write the estimate down, then build the thing and check yourself against reality. The candidate who says "I measured this and got 63 GB/s, not 64, because of line encoding" is unmistakably different from one who recites numbers.

---

## 3. Portfolio artifacts — what actually works

Hiring managers cannot evaluate skills from a resume. They evaluate **evidence**. Ranked by how much they move the needle:

**Tier 1 — decisive**

- **A public benchmark report with your own numbers.** Real hardware, stated method, warmup, repetitions, a plot, and an honest interpretation. "I served a 7B at concurrency 1-512 and here is the knee, and here is why" beats a certificate.
- **An optimization writeup with before/after and a root cause.** Not "I used FP8" but "decode was bandwidth-bound; quantizing the KV cache cut bytes/token enough to raise batch occupancy, worth 1.7x at fixed p95."
- **A merged contribution to a real AI-infra project.** See Section 5 — but note the bar has risen.
- **A reproducible repo for one of the [projects](PROJECTS.md).** With a README containing the numbers and what surprised you.

**Tier 2 — strong supporting evidence**

- **A failure post-mortem you wrote.** "Here is how I found the straggler rank" demonstrates diagnostic skill better than a success story.
- **A cost analysis that changed a decision.** $/1M tokens, assumptions stated, and the action taken.
- **A design doc** for something you scoped — including what you chose not to build.

**Tier 3 — weak but not worthless**

- Reading notes, course completions, certificates, curated awesome-lists. These show interest, not ability.

**What does NOT work**

- **Tutorial replications with no deviations.** Everyone runs the same vLLM quickstart. Nobody learns anything from yours.
- **A list of 40 tools you have "used."** Depth in one layer is worth more than breadth of installation.
- **Model fine-tunes with no evaluation.** "I fine-tuned Llama" without a measured delta is unverifiable and reads as a tutorial.
- **Claims you cannot defend.** Any number you put in public is a number you will be asked to reproduce. Over-claiming is worse than a modest, solid result — and it is the most common failure mode.
- **Private work described but not shown.** Describe the *method and reasoning* (sanitized) rather than the confidential result.

**The evidence discipline that prevents over-claiming.** Tag your own work honestly:

| Tier | What it means | Example phrasing |
|---|---|---|
| **Measured** | You ran it, have the config, and it reproduces | "Measured 1,840 tok/s at p95 TTFT 410 ms on 1xH100" |
| **Estimated** | You derived it from first principles, not measured | "Predicted ~140 GB from the memory formula; did not verify on hardware" |
| **Read** | You studied it but did not run it | "I understand the design; I have not deployed it" |
| **Heard** | Second-hand, unverified | "I have read that this scales, but cannot confirm" |

Keeping these straight in your own head is the difference between a candidate who is trusted and one who is not. Interviewers probe; the ones who say "I estimated that, I did not measure it" gain credibility, and the ones who bluff get caught on the next question.

---

## 4. Breaking in from adjacent backgrounds

| You are | Your real advantage | The bridge work |
|---|---|---|
| **Backend / full-stack** | You already debug distributed systems, containers, and APIs | Do Stage 0-2 properly. Your gap is numerics (bytes, precision, roofline) and GPU profiling — not software engineering |
| **Data engineer** | Pipelines, storage, and scale are your home turf | Go deep on L4 + Stage 2 serving. Learn why training I/O is different (many-small-files, sharded formats, checkpoint bandwidth) |
| **SRE / ops** | You think in SLOs, incidents, and failure modes — this is the scarcest mindset in AI infra | Learn the GPU stack and the serving/training internals. Your reliability instincts are the differentiator, not the gap |
| **Data scientist / ML engineer** | You know models, evaluation, and the math | Your gap is systems: containers, collectives, profiling, cost. Resist the urge to stay in notebooks |
| **Kernel / HPC / graphics** | CUDA, memory hierarchies, performance are already yours | Learn the ML workload shapes and the serving stack. This is one of the fastest transitions available |
| **New grad** | Time, and no unlearning to do | Build 3-4 Tier-1 artifacts *before* applying. Stage 0 is not optional, and a single deep project beats six shallow ones |
| **Self-taught** | You have already proven you can learn without a syllabus | You need *proof*, not credentials. Public benchmark reports are the great equalizer |

**The universal bridge:** pick one layer, build the artifact, write it up publicly, and repeat. There is no shortcut that substitutes for having done the work, but there is also no credential that substitutes for it either.

---

## 5. Contributing to open source (the bar has risen)

"Fix a typo to get your first PR" is **dead advice**. Major projects have explicitly tightened rules because of low-effort and AI-generated drive-by contributions:

- Large serving projects **cap the number of open PRs** for contributors without write access, and have moved to **banning purely agent-generated PRs**.
- Some maintainers now **require disclosure of AI assistance** and human review of every line.
- Projects have merged policy PRs stating that **new contributors should not submit trivial fixes**.

**What actually works:**

1. **Use the project seriously first.** Run it, hit a real wall, and fix that. Maintainers can tell the difference instantly.
2. **Start with reproduction and triage, not code.** Confirming a bug with a minimal repro, adding a regression test, or improving a confusing error message is genuinely valuable and low-risk.
3. **Read `CONTRIBUTING.md` and the CI setup before you write anything.** Match the house style; run the linters locally.
4. **Do one thing well in one project** rather than drive-by PRs across ten.
5. **Engage in issues and discussions first.** Getting known as someone who gives good bug reports is how you get invited to do more interesting work.

**The realistic path:** reproduce a benchmark → find the thing the docs get wrong → fix the docs *and* the code → maintain it. Most long-term contributors started by being useful in an issue thread, not by opening a PR on day one.

**Where to look:** the projects in [STACK.md](STACK.md) are all real and most take contributions. Smaller projects (the L6-L8 layers, and the newer RL post-training tools) have far less competition for meaningful work than vLLM or PyTorch.

---

## 6. Compensation and leveling

> **Read this as structure, not data.** 2026 public compensation data for AI infra is thin and self-reported. Ranges vary by 2-3x for the same title depending on region, company stage, and whether equity is liquid. **Do not anchor a negotiation on numbers from a roadmap.**

What *is* reasonably stable:

- **Structure:** base + bonus + equity. At startups, equity is usually the majority of the upside and the majority of the risk. Ask what the strike price, preferred valuation, and liquidity path are.
- **Leveling:** most large companies map to a common ladder (roughly: new grad → mid → senior → staff → principal). Scope, not title, is what transfers between companies. Ask "what decisions does this level own?"
- **Differentials:** performance/kernel specialists and people who have carried a production SLO tend to command a premium over generalist platform roles at the same level, because the supply is smaller.
- **How to get real numbers:** ask people in the role, use levels-style aggregators with skepticism, and weight *recent, local* data. Compare total compensation, not base.

**The honest strategic advice:** early in your career, optimize for the work and the mentor, not the delta in offer. The compounding from owning something real at scale dwarfs a 10% base difference — and it is what lets you negotiate from strength two roles later.

---

## 7. What employers actually screen for

**On a resume/GitHub:**

- **Evidence of having run something, with numbers.** Links to writeups beat bullet points about responsibilities.
- **Depth in one area plus adjacent awareness.** The "one layer deep" shape.
- **Debugging narratives.** "Reduced cost 40% by finding that X" tells a story; "worked on inference optimization" does not.
- **Ownership signals.** SLOs carried, incidents run, systems still running.

**Common mistakes:**

- Listing tools instead of outcomes.
- No links to anything runnable or readable.
- Claims that collapse under one follow-up question.
- Ignoring the *soft* half — the job is largely convincing other teams to adopt what you built.

**In the interview:**

- Guessing instead of stating assumptions.
- Silently computing (narrate your reasoning).
- Defending a design after new information contradicts it.
- Not asking clarifying questions — the workload *is* the question.

---

## 8. A realistic 12-week plan

Assumes ~10 hours a week, and that you already write code.

| Weeks | Do | Artifact |
|---|---|---|
| 1-2 | Stage 0 labs. Learn your machine's real numbers | A bytes-and-bandwidth memo with your hardware's measured figures |
| 3-4 | Stage 1 labs 1-3. Train small, fine-tune, serve locally | A fine-tune with an eval delta; a tok/s table at three context lengths |
| 5-6 | Stage 1 labs 4-6 + Stage 2 labs 1-2. Profile and benchmark | A profile with a real speedup; a latency-throughput curve |
| 7-8 | Stage 2 labs 3-6. Optimize to an SLO | A $/1M-token report and a one-page SLO memo |
| 9 | Stage 2.5 labs 1, 3, 5 (RL, cheap versions) | A `T_gen`/`T_train`/`T_sync` breakdown and a length histogram |
| 10 | Stage 3 labs 1-2 on spot instances | An `nccl-tests` report and a parallelism plan |
| 11 | Write it up. One public post with real numbers and real failures | The post *is* the artifact |
| 12 | Apply. Start contributing to one project you actually used | 3-5 applications + one issue triaged properly |

**Two things that make this work:** publish as you go rather than at the end (momentum and feedback), and keep every number reproducible — you will be asked.

---

**Back to:** [README](README.md) · [ROADMAP](ROADMAP.md) · [PROJECTS](PROJECTS.md) · [RESOURCES](RESOURCES.md)
