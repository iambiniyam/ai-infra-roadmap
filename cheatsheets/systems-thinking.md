# Systems Thinking Cheatsheet

The four laws that govern every serving and training fleet, and where the classic theory breaks for LLMs. Formulas here are durable; they will outlive every model and every GPU in this repo.

---

## 1. Little's Law — the one you use every week

```text
L = λ × W

L = number of items in the system (concurrency, in-flight requests, jobs)
λ = arrival rate (requests/sec, jobs/sec)
W = mean time each item spends in the system (latency)
```

**Why it is special:** it is distribution-free. It does not assume Poisson arrivals, exponential service, or independence. It requires only that the system be stable and the averages exist. That makes it the most robust tool here, and it is why the naive queueing models break for LLMs but this does not.

**The three forms you actually need:**

| I want to know | Solve for | Formula |
|---|---|---|
| How much concurrency will this SLO imply? | `L` | `L = λ × W` |
| What throughput can this fleet sustain? | `λ` | `λ = L / W` |
| What latency will this load produce? | `W` | `W = L / λ` |

**Worked example — sizing a serving fleet.** Suppose a chat product takes 50 req/s and you have promised a 2 s end-to-end mean latency:

```text
L = 50 req/s × 2 s = 100 concurrent requests
```

So you must be able to hold **100 requests in flight at once**. If your engine handles 40 concurrent sequences per GPU at your context length (KV-cache bound — check the formula in [inference-math.md](inference-math.md)), you need at least ⌈100/40⌉ = **3 GPUs** before any headroom.

**But stop there and you will be under-provisioned**, because Little's Law gave you the *mean*. Apply headroom sized by the variance (Section 2) and the bursts (Section 5):

```text
raw replicas           = 100 / 40                 = 2.5  → 3
at a 65% operating point = 2.5 / 0.65             = 3.85 → 4
at peak-minute PMR = 8.0 = 2.5 × 8 / 0.65         = 30.8 → 31
```

**The gap between 3 and 31 is the lesson of this whole cheatsheet.** The mean requirement is 3; the peak-minute requirement is 31. You cannot statically provision for the peak (unaffordable) and you cannot survive provisioning for the mean (you fall off the utilization cliff). That gap is exactly why admission control and autoscaling are not optional — Sections 2 and 5.

**The reverse use is the production workhorse.** If you observe 8 replicas each averaging 22 in-flight requests and a measured completion rate of 70 req/s:

```text
L = 8 × 22 = 176 requests
W = L / λ = 176 / 70 = 2.51 s
```

You just derived end-to-end latency from concurrency and throughput with **no tracing at all**. If it contradicts your SLO, you know before a user tells you.

**The intuition:** concurrency is not a tuning knob you pick; it is *induced* by your arrival rate and latency target. Setting `--max-num-seqs` below the implied `L` means queueing. Setting it far above wastes KV cache.

---

## 2. The utilization trap — why 100% is a bug, not a goal

Queueing delay does not grow linearly with utilization; it grows hyperbolically.

```text
M/M/1 mean response time:   W = (1/μ) / (1 − ρ)
                            ρ = λ/μ = utilization

W / (1/μ) = 1 / (1 − ρ)     ← the multiplier over service time
```

| Utilization ρ | Mean wait × service time |
|---:|---:|
| 0.50 | 2.0x |
| 0.70 | 3.3x |
| 0.80 | 5.0x |
| 0.90 | 10.0x |
| 0.95 | 20.0x |
| 0.99 | **100x** |

**Read the last row again.** At 99% utilization the average request waits 100x its own service time. This is why "our GPUs are 95% utilized" is usually a description of a system with terrible tail latency, not an efficient one.

**The design consequence:** you *choose* your utilization from your tail-latency target, not the reverse.

- A latency-critical interactive service (p99 matters) should target roughly **50-70%** per-server utilization.
- A batch or offline workload (throughput matters, latency does not) can run at **90%+**.
- This is exactly the tradeoff in [HARDWARE.md](../HARDWARE.md) between MFU for training (~40-55% is good) and *median production utilization of ~50%* — the idle headroom is a feature you are buying with money, and it is what absorbs bursts.

**Where the classic theory breaks for LLMs.** M/M/1 assumes memoryless service. LLM service times are **not** memoryless — they depend on output length, which is heavy-tailed, and on batch composition, which the scheduler controls. So use the *shape* of the curve (hyperbolic blow-up near ρ→1) as design intuition, not the exact multiplier. The qualitative rule — **leave headroom, or pay in the tail** — survives; the specific numbers do not.

### Quantifying the tail: the Pollaczek–Khinchine factor

To go beyond "heavy tails are bad," add the variance term:

```text
W_q = E[S] · ρ/(1−ρ) · (1 + C_s²)/2

C_s² = Var(S) / E[S]²   ← squared coefficient of variation of service time
```

That `(1 + C_s²)/2` factor is the whole point: **variability costs latency exactly as much as utilization does.**

| `C_s²` | Regime | Multiplier `(1+C_s²)/2` |
|---:|---|---:|
| 0 | deterministic | 0.5 |
| 1 | exponential (the M/M/1 assumption) | 1.0 |
| 2-8 | **realistic LLM output lengths** | **1.5-4.5** |
| 16 | agentic / long-CoT mixture | 8.5 |

**Worked with realistic numbers.** Measured output lengths over 1,000 requests: 60% short (40 tok), 30% chat (300), 8% RAG (1,200), 2% agentic (6,000):

```text
mean = 330 tokens        p50 = 40 tokens        mean/median = 8.25x
Var  = 754,260           C_s² = 754,260 / 330² ≈ 6.93
```

At ρ = 0.7 with `E[S] = 50 ms`:

```text
M/M/1 assumes:      W_q = 50 × 2.333               =  117 ms
The real tail:      W_q = 50 × 2.333 × (1+6.93)/2  =  463 ms
```

**The same utilization costs ~4x more latency purely because of the distribution.** And note what the mean hides: **2% of requests generate 36% of all output tokens.** Sizing capacity on mean output length under-provisions badly.

**The actionable consequence:** reducing `C_s²` — via length-aware scheduling, admission control, or disaggregating by request class — is often **cheaper than adding GPUs**, and it is the same lever that makes RL rollouts efficient (see [RL post-training](rl-post-training.md)).

**The generalized form (Kingman / VUT)** adds arrival variability:

```text
E[W_q] ≈ E[S] · (ρ/(1−ρ)) · (C_a² + C_s²)/2
          ↑         ↑              ↑
        time   utilization    variability
```

**Wait = Variability × Utilization × Time.** Three levers — and most teams only ever pull the middle one.

**Where these models fail for LLMs — state this, do not paper over it:**

1. **Service time is state-dependent.** P-K assumes i.i.d. service independent of queue state. In an LLM server, adding a request to a batch raises per-token latency for *everyone* (jobs are coupled) while simultaneously raising throughput per GPU-second. That second effect is a *negative feedback* classical queueing cannot express. **So M/G/1 tends to over-predict `W_q`; treat it as an upper bound, not a forecast.**
2. **The bottleneck resource changes between jobs.** Prefill is compute-bound; decode is memory-bandwidth-bound. That is a *mixture of two servers with different bottlenecks*, and the mixing ratio depends on the scheduler. Classical queueing has no vocabulary for it.
3. **Continuous batching re-forms the batch every iteration.** Requests join and leave mid-flight — neither non-preemptive nor a standard preemption discipline.
4. **Agentic arrivals are endogenous.** Turn *n+1* is produced by turn *n*, so λ is not an exogenous Poisson input. Little's Law still holds (it needs only ergodicity); any model assuming independent arrivals is structurally wrong.

**The lesson in point 2 is worth pausing on: the systems response to a queueing-model failure was to change the architecture until the model applied again.** DistServe and Sarathi-Serve exist precisely because prefill and decode could not be modeled as one server.

**The LLM-specific version of this** is the goodput curve from [inference-math.md](inference-math.md): as you raise concurrency to increase throughput, per-request latency rises, and past the knee you gain little throughput while destroying the SLO. Find the knee; that is your operating point.

---

## 3. Amdahl's Law — the ceiling on any optimization

```text
Speedup = 1 / ( (1 − p) + p/s )

p = fraction of total time the optimized part accounts for
s = speedup of that part alone
```

The crucial term is `(1 − p)` — the part you *did not* touch. It caps everything.

| Optimized fraction `p` | Even at `s = ∞` | At `s = 2x` |
|---:|---:|---:|
| 10% | 1.11x | 1.05x |
| 25% | 1.33x | 1.14x |
| 50% | 2.00x | 1.33x |
| 75% | 4.00x | 1.60x |
| 90% | 10.00x | 1.82x |

**Worked example — the edge/cloud placement case.** A task uploads 30 MB at 20 Mbit/s, computes 0.3 s, and downloads 5 MB at 100 Mbit/s, with 0.1 s RTT:

```text
T = 30×8/20 + 0.1 + 0.3 + 5×8/100 = 12.0 + 0.1 + 0.3 + 0.4 = 12.8 s

Compute is 0.3/12.8 = 2.3% of the task.
Make the model 10x faster → T = 12.53 s. You saved 0.27 s (2%).
Even at infinite speed → T = 12.5 s.
```

**The network is 94% of the task, so optimizing the model is pointless.** Halving the *input* instead (a 0.15 s codec) saves 6 s — a 22x better use of effort. This is the honest argument for edge placement: once transfer dominates, model efficiency stops mattering, and the only wins left are moving less data or moving it a shorter distance.

**Amdahl's corollary for infra:** measure the profile *before* choosing what to optimize. A 10x kernel speedup inside a 10% stage is a 1.05x result, and the engineer who did it will struggle to explain why the dashboard did not move.

**Gustafson's Law is the companion:** Amdahl bounds speedup at *fixed problem size*; Gustafson notes that in practice you scale the problem with the machine, so the achievable speedup grows with the parallel portion. This is why "we added GPUs and trained a bigger model" is a legitimate strategy while "we added GPUs and ran the same job faster" hits Amdahl's wall.

---

## 4. Critical path — who must wait

Little's Law and Amdahl both concern *rates and fractions*. The critical path concerns **dependencies**: the longest chain of steps that must happen in order.

```text
T_min = Σ (duration of each step on the longest dependency chain)
```

Everything off the critical path is potentially free to overlap. Everything on it is a hard floor.

**Where this beats the roofline.** The roofline tells you the best possible time for a *single* operation. It says nothing about a pipeline. For multi-stage and multi-turn workloads — disaggregated serving, agent loops, training with pipeline parallelism — the critical path is the right tool:

| Workload | What bounds it |
|---|---|
| Single kernel, batch 1 | Roofline (memory term, usually) |
| Serving at high concurrency | Throughput + queueing (Sections 1-2) |
| Multi-turn agent | **Critical path** — every tool call is a serial round trip |
| Pipeline-parallel training | **Critical path** — the slowest stage sets step time (plus bubbles) |
| Disaggregated prefill/decode | **Critical path** — the KV transfer is serial between stages |

**Agent workloads are the clearest case.** A 40-turn agent has 40 sequential model→tool→model round trips. If each turn costs 2 s — 1.5 s of inference and 0.5 s of network — then `T = 40 × 2 = 80 s`, made up of 60 s of inference and 20 s of network. The turns are serial, so:

- Doubling inference throughput saves at most 30 s (and only the inference part).
- The 20 s of network is **structurally unreachable** by any inference optimization.
- The only way to remove it is to reduce the *number of round trips* (fewer, larger tool calls) or co-locate the tool.

**Optimize turn count and per-turn latency, not tokens/sec.**

---

## 5. Workload characterization — means are lies

Every formula above takes a **mean**. Real workloads are heavy-tailed, and for tail-sensitive systems the mean is the wrong summary.

**What to measure and report:**

| Statistic | Why |
|---|---|
| p50 / p95 / p99 / max | The tail is what users experience and what SLOs measure |
| Distribution shape (histogram), not just moments | A bimodal latency distribution means two different code paths, not noise |
| Peak-to-mean ratio | Sizes your headroom; a 3:1 burst means capacity planning at the mean fails |
| Coefficient of variation (σ/μ) | > 1 signals heavy tail; queueing formulas assuming exponential are unsafe |

**Why LLM workloads are unusually heavy-tailed:**

- **Output length** is right-skewed and unbounded. One request can take 10x the median. This is why [RL post-training](rl-post-training.md) has skewness bubbles and why serving has a latency tail.
- **Request arrival** is bursty, not Poisson — traffic follows human rhythms and upstream retries.
- **Per-request cost** varies with both input and output length, and they are independent.

**The trap:** provisioning from mean latency and mean throughput. If your fleet is sized so that the *mean* load fills it, then any burst — or any single long request — pushes you past ρ = 0.9 and into the Section 2 cliff. **Size for the peak you must survive, and state which percentile you provisioned for.**

---

## 6. Measurement traps: how to fool yourself

Most "we made it 3x faster" claims are measurement artifacts. These are the specific ways it happens, each with the fix:

| Trap | What goes wrong | The fix |
|---|---|---|
| **CUDA asynchrony** | You time a kernel launch, not the kernel. Python returns immediately; the work has not started | `torch.cuda.synchronize()` before stopping the timer |
| **No warm-up** | First iterations include JIT compilation, allocator growth, clock ramp. Including them can *flatter or ruin* your result depending on which side you measure | Discard the first N iterations, and say what N was |
| **Profiler overhead** | `nsys`/`torch.profiler` can slow the workload 2-10x and shift which kernel dominates | Use it to find *relative* shares, not absolute times |
| **Lost in the trace** | Thousands of events make the real cost invisible | Filter with NVTX ranges around what you care about |
| **DVFS / thermals** | The GPU clocks down under sustained load; a short benchmark runs at boost | Run long enough to reach steady state; report clocks |
| **FP non-associativity** | Different reduction order gives different results run to run | Compare with a tolerance and a seed, not `==` |
| **GIL-bound load generator** | Your Python client saturates one core and becomes the bottleneck, so you measure the client, not the server | Check client CPU; use a compiled/async generator |
| **Ignoring EOS** | "max_tokens" is a *cap*, not a promise. Fixed-length benchmarks overstate throughput because real traffic stops early | Use `min_tokens`/`ignore_eos` for throughput runs, and say so |
| **Synthetic uniformity** | A load generator with `stddev=0` produces a workload no user ever creates: no long tail, no burstiness | Generate from a measured distribution, or replay a real trace |
| **Client-side queueing** | At high concurrency, requests queue in the *client* before reaching the server | Measure arrival at the server, not departure from the client |

**The meta-trap: Goodhart's Law.** Once a metric becomes a target, it stops measuring what you wanted. Throughput at the cost of the p99 SLO; MFU at the cost of correctness; eval score at the cost of generalization. **Always pair a metric with the constraint it must not violate.** "1,800 tok/s *at p95 TTFT < 500 ms*" is a real result; "1,800 tok/s" is a number looking for a way to mislead you.

**The checklist before you believe any number:**

- **Warmup:** discarded, and you said how much.
- **Repetitions:** ≥5, with the spread reported, not one run.
- **The noise floor:** run the *same* config twice. The gap is the smallest change you can detect. Any improvement below it is not an improvement.
- **Isolate variables:** change one thing. If you changed two and it got faster, you learned nothing about either.
- **State the conditions:** shared GPU? thermal state? which clocks? These move results by tens of percent.
- **Name the constraint:** what must this not break?
- **Report negative results.** "I tried X and it made things worse" is data. The best sources in [RESOURCES.md](../RESOURCES.md) publish their misses; so should you.

> **The highest form of this discipline is a "recorded miss."** An honest benchmark harness states which of its own targets it failed. A source that only publishes wins is telling you about its marketing, not its method.

---

## 7. The back-of-envelope checklist

Before any design discussion, produce these numbers on paper:

1. **What must move**, how much, how many times, through which interface? ([the five questions](../HARDWARE.md#0-the-five-questions-start-here-every-time))
2. **Iron Law:** `T = D/BW + O/(R·η) + L` — which term dominates, and by how much?
3. **Capacity:** does it fit? (weights + KV + activations + workspace)
4. **Bandwidth:** can the interface supply the data at the rate compute demands?
5. **Little's Law:** at my arrival rate and latency target, what concurrency is implied?
6. **Utilization:** what ρ does that concurrency imply, and what does Section 2 say about the tail?
7. **Critical path:** what must wait, and is it on the path or overlappable?
8. **Amdahl:** what fraction is the thing I am about to optimize?

If you can answer all eight in ten minutes, you can rule out most bad designs before writing code — and you will be the person in the room who knows which number matters.

---

## 8. Estimation drills (build the habit)

Estimation is a trainable skill and it is trained by being wrong repeatedly, in private, where it costs nothing. Work these on paper, then check.

| # | Estimate | The trap it exposes |
|---|---|---|
| 1 | How many concurrent requests does 40 req/s at 2.5 s imply? | Confusing rate with concurrency |
| 2 | A 70B bf16 model at batch 1 on an H100 — tok/s? | Forgetting the weight read happens *per token* |
| 3 | Same model, batch 64 — tok/s? | Assuming linear scaling past the roofline knee |
| 4 | KV cache for 32k context × 64 concurrent, 70B GQA? | Forgetting the `2 ×` for K and V, or using query heads instead of KV heads |
| 5 | Time to upload 1 GB over a 1 Gbit/s link | Bits vs bytes (the classic 8x error) |
| 6 | Time to read a 10 GB checkpoint from local NVMe | Confusing GB/s with Gb/s, or forgetting storage is shared |
| 7 | Does 8× tensor parallelism give 8× speedup on a PCIe-only node? | Ignoring that the interconnect, not the compute, is the limit |
| 8 | Fleet size for 100 req/s at a 2 s SLO, 16 concurrent/replica, 70% target | Forgetting headroom entirely |
| 9 | All-reduce time for a 70B gradient across 16 nodes over IB NDR | Not knowing the collective's data volume (2(N−1)/N × size) |
| 10 | $/1M tokens for a 7B at 2,000 tok/s on a $2/hr GPU | Omitting utilization |

**Score yourself: within 2x is competent, within 20% is good.** For every miss, write down *which term you forgot*. That list is your actual curriculum — it will look like "forgot KV," "forgot that it repeats per token," "assumed peak bandwidth."

**Then verify three of them on real hardware.** The gap between your estimate and the measurement is where the learning is; a clean match teaches you nothing new.

---

## 9. Free resources

| Resource | Why |
|---|---|
| [Machine Learning Systems (CS249r)](https://mlsysbook.ai/) | The Iron Law and its application, taught at book length with modelling tools |
| [AI Infra in Depth](https://github.com/bojieli/ai-infra-book) | Quantitative derivation of inference and training systems from constraints |
| [Google SRE: Handling Overload](https://sre.google/sre-book/handling-overload/) | Load shedding, admission control, and why you cap concurrency |
| [Google SRE Workbook: Managing Load](https://sre.google/workbook/managing-load/) | Capacity planning and demand forecasting in practice |
| [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/) | Why p99 matters in fan-out systems, and the mitigations |
| [Little's Law (Wikipedia)](https://en.wikipedia.org/wiki/Little%27s_law) | The statement and its (weak) assumptions |
| [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law) · [Gustafson's Law](https://en.wikipedia.org/wiki/Gustafson%27s_law) | The two limits on parallel speedup |
| [Kingman's Formula](https://en.wikipedia.org/wiki/Kingman%27s_formula) | The VUT approximation behind Section 2; accurate near saturation |
| [Pollaczek–Khinchine Formula](https://en.wikipedia.org/wiki/Pollaczek%E2%80%93Khinchine_formula) | Where the `(1+C_s²)/2` variance term comes from |
| [DistServe (OSDI '24)](https://arxiv.org/abs/2401.09670) · [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | What it looks like when a queueing failure forces an architecture change |
| [Orca (OSDI '22)](https://www.usenix.org/conference/osdi22/presentation/yu) | Iteration-level scheduling; why classical batch/service assumptions stop holding |
| [Harchol-Balter, CMU](https://www.cs.cmu.edu/~harchol/) | The standard queueing textbook for computer systems |
| [Systems Performance (Gregg)](https://www.brendangregg.com/systems-performance-2nd-edition-book.html) | The USE method and measurement methodology |

---

**Back to:** [README](../README.md) · [ROADMAP](../ROADMAP.md) · [HARDWARE](../HARDWARE.md) · [inference math](inference-math.md) · [RL post-training](rl-post-training.md) · [eval math](eval-math.md)
