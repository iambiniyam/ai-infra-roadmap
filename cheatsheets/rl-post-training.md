# RL Post-Training Cheatsheet

The systems half of RL post-training: where the time goes, why it does not scale like pretraining, and the four invariants worth memorizing. Pair with [inference-math.md](inference-math.md) (KV cache) and [training-parallelism.md](training-parallelism.md) (sharding).

---

## 1. Why RL is a different systems problem

Pretraining is **one** tightly-coupled collective job: every FLOP produces a gradient. RL post-training is **two** loosely-coupled workloads in a loop, with different bottlenecks:

| Stage | Bound by | Scales with |
|---|---|---|
| **Rollout / generation** | HBM **bandwidth** | aggregate memory bandwidth (chiefly TP) |
| **Training / backprop** | **Compute** (FLOPs) | more GPUs, almost linearly |
| **Weight sync (reshard)** | **Interconnect** + memory | how clever your reshard is |

Plus a third tier in agentic RL: **verifiers and sandboxes**, which are services with latency and a security boundary.

> **Invariant 1: rollout dominates wall-clock.** Generation typically consumes the majority of RL training time, and its share *grows with output length*. Long-CoT reasoning models make it worse by design — the better RL works, the longer the model thinks.

---

## 2. The skewness bubble (the root cause)

Output lengths are right-skewed, not normal. Decoding is bandwidth-bound, so throughput depends on keeping the batch large. But late in a generation round only a few long samples remain, batch size collapses, and the GPU idles. This is the **skewness bubble**.

Consequences you must engineer around:

- **Group completion is a max, not a mean.** Group-relative methods (GRPO/DAPO) cannot score a group until *every* sample in it finishes. So the **slowest trajectory in each group sets the pace**.
- **Therefore: shrink P99, not the mean.** Length-aware scheduling, priority scheduling, and skewness-aware dispatching all exist for this reason.
- **Optimizing per-GPU utilization can *lower* end-to-end throughput** by scattering rollout progress across many groups and delaying the completion of enough groups to unblock the next optimizer step. This is counterintuitive and it is real.

```text
Effective throughput  ≈  groups completed per unit time
                      ≈  B_groups / max_over_groups( Σ_turns T_turn )

The max() is why agentic RL ≠ stateless serving.
```

---

## 3. The iteration-time equation

```text
T_iter = max( T_gen(x), T_train(y) )    subject to  x + y ≤ N GPUs
```

Minimize the **maximum** of the two stages. Because `T_gen` *saturates* as you add GPUs (scaling out generation mainly adds data-parallel replicas, and only TP increases aggregate bandwidth — and TP is capped by intra-node NVLink) while `T_train` keeps improving, **the optimum is usually not 50/50**, and it shifts *during training* as outputs get longer.

**Practical reading:** allocate generation roughly in proportion to the fraction of iteration time it consumes. If rollout is 70% of wall-clock, a balanced split badly under-provisions generation.

**And budget the reshard.** In a colocated setup, weight sync is serial and sits on the critical path:

```text
T_iter = T_gen + T_train + T_sync      (colocated, serialized)
T_iter = max(T_gen, T_train)           (fully overlapped / async)
```

---

## 4. Weight resharding — what it actually is

The same weights must exist in two **incompatible parallel layouts**:

- **Training layout:** FSDP/ZeRO shards parameters across DP ranks, or Megatron shards by TP×PP×EP. Each GPU holds a *slice*.
- **Inference layout:** vLLM/SGLang want a different TP degree for KV-cache distribution, and may replicate weights for fast decode.

**Resharding is the all-to-all redistribution between them, every iteration.** The naive implementation round-trips through a fully-materialized copy (often via CPU or disk), which costs O(model size) in both memory and bandwidth and forces you to have room for *both* layouts at once.

**Why it matters more than people expect:** teams carefully tune `T_gen` and `T_train`, then lose a large fraction of iteration time to an unoptimized weight sync. Slower-than-expected RL iterations are more often reshard-bound than compute-bound. Mitigations: fused resharding engines (no full materialization), **delta weight sync** (send only changed bytes), and disk/storage-mediated sync for cross-fleet rollout.

> **Invariant 2: the reshard is on the critical path.** Measure `T_sync` separately or you are debugging blind.

---

## 5. Colocated vs disaggregated — the answer flipped twice

Worth learning as a case study in *why* architecture opinions reverse.

| Era | Design | Why it won / lost |
|---|---|---|
| Original | **Disaggregated** — separate GPUs for generation and training | Simple, reuses inference stacks. **Flaw:** each side idles while the other runs |
| ~2024 | **Colocated** — same GPUs, time-multiplexed; training state parked in CPU memory | Killed the idling. Became dogma |
| 2025+ | **Disaggregated, revisited** | Colocation forces **resource coupling** |

**The real argument for disaggregation is resource coupling, not raw speed.** Generation is bandwidth-bound; training is compute-bound. Colocation forces both stages onto identical GPU counts and identical hardware. The consequence: generation saturates while training keeps scaling, so you systematically over-provision and under-use the rollout fleet.

The economics make it concrete — an inference-optimized part can have *more* memory and bandwidth at a fraction of the cost of a training part. Colocation forbids using it, because your trainer needs the compute. Disaggregation lets each stage buy the right silicon.

Two supporting facts:

- Disaggregated RL needs only **point-to-point** transfer (samples out, weights back), not pretraining-style full-mesh collectives. That makes **cross-datacenter and heterogeneous-fleet RL genuinely feasible**.
- **Async removes weight sync from the critical path entirely.** One-step asynchronous RL — generate with Wᵢ, train on samples from Wᵢ₋₁ — is empirically reported not to measurably hurt convergence. Beyond one step you need staleness correction (importance sampling), and heavy-tailed importance ratios inflate gradient variance.

**Degrees of asynchrony**, as a ladder:

1. **Colocated sync** — global barrier each iteration.
2. **Mini-batch pipelining** — training starts once the first mini-batch lands. The long tail means the last mini-batches still spill over.
3. **One-step async** — dependency shifts across iterations; weight sync leaves the critical path.
4. **Stream generation** — send each sample to training the instant it completes, enabling sample-level reference/KL/reward.

**Rule of thumb:** colocation is fine below roughly 100 GPUs on homogeneous hardware. Disaggregation wins at scale, with heterogeneous GPUs, or when the two stages want very different parallel configs.

---

## 6. Memory math — why GRPO changed the cluster

Per-parameter training memory (Adam, mixed precision, unsharded):

```text
bf16 param 2 + bf16 grad 2 + fp32 master 4 + Adam m 4 + Adam v 4  ≈ 16 bytes/param
```

**PPO** keeps four models resident: actor, critic, reference (frozen), reward.
**GRPO** drops the critic and — with programmatic verifiers — the reward model, keeping actor + reference.

| | PPO | GRPO |
|---|---|---|
| Resident models | 4 | 2 |
| First-order training memory | ~4x | **~2x** |

Treat "4x vs 2x" as a first-order approximation, not a constant (the reference model is frozen and needs no optimizer state, so it costs ~2 bytes/param, not 16).

**Why this is an infrastructure result, not just an algorithm result:** halving resident training memory is what makes colocation feasible at all, because the rollout engine's weights *and* KV cache must fit in the memory the trainer is not using. GRPO is the reason 7B-32B RL fits on 8 GPUs.

---

## 7. Rollout KV cache

Same formula as serving, but the numbers bite harder because rollout runs huge batches at long context:

```text
KV bytes/token = 2 (K and V) × n_layers × n_kv_heads × head_dim × bytes_per_value
KV_total       = concurrent_sequences × mean_length × KV bytes/token
```

`n_kv_heads` is the **KV** head count under GQA, not the query head count. That is the single most common error.

**Worked example** (70B-class, 80 layers, 8 KV heads, head_dim 128, bf16):

```text
per token   = 2 × 80 × 8 × 128 × 2 = 327,680 B ≈ 0.31 MiB
16k tokens  ≈ 5.0 GiB per sequence
128 concurrent sequences ≈ ~640 GiB of KV
```

That is the whole long-CoT bottleneck in one number: on an 80 GB GPU you hold a handful of such sequences **before** weights, and since decode throughput scales with batch size, you are forced to trade context against throughput — and the long tail gets the worst of it.

This is also why generation **saturates**: once bandwidth-bound, tokens/sec depends on memory bandwidth, not on concurrency.

---

## 8. Metrics that actually tell you the truth

| Metric | Why |
|---|---|
| `T_gen` / `T_train` / `T_sync` split | The only way to find where an RL step goes. Instrument all three. |
| **Group-completion throughput** (groups/hour) | The agentic-RL analogue of goodput. Not MFU. |
| **P50 vs P99 trajectory length/latency** | The gap *is* the skewness bubble you must engineer around |
| Generation time fraction vs output length | Should climb; if it does not, your task is not producing long CoTs |
| Peak memory during rollout | KV cache, not weights, is usually the binding constraint |

> **Invariant 3: measure the three stages separately.** `T_gen`, `T_train`, and `T_sync` must be distinguishable in your dashboards, or every optimization is a guess.

---

## 9. Verifiers and sandboxes as infrastructure

- **RLVR (verifiable rewards)** — programmatic/executable checks for objectively-checkable domains (math, code) — has displaced learned reward models for a large share of work.
- The verifier is a **service**, executed in a **sandbox** (Docker/microVM/Firecracker), and its latency is on the critical path. **Verifier throughput can bottleneck RL just as rollout can.**
- **Sandboxes must autoscale.** Statically-provisioned sandboxes become the bottleneck under load.
- **Sandbox isolation is a security boundary, not a nicety.** Agentic RL executes model-generated code. microVM-class isolation is not paranoia.

**Multi-turn changes the unit of work** from a generation to a **trajectory** (a sequence of observations/actions across tool calls). That means new infrastructure: long-lived stateful environment servers, trajectory storage (not just token sequences), tool sandboxing, and the group-completion metric above.

---

## 10. Traps

1. **Treating "rollout is ~90% of training time" as a planning constant.** The *direction* is well corroborated; the *magnitude* varies enormously with model, task, and max length. Measure yours.
2. **Assuming colocation is universally right — or universally dead.** It flipped twice. The answer is `max(T_gen, T_train)` minimization, and it is workload-dependent.
3. **Forgetting the reshard.** The most under-taught cost in RL infra.
4. **Optimizing MFU in agentic RL.** Group completion is a max; per-GPU utilization can anti-correlate with throughput.
5. **Assuming off-policy is free.** One step is reportedly safe; beyond that, correct the staleness.
6. **Ignoring the long tail.** RL actively makes it worse by teaching longer reasoning.
7. **Treating sandboxes as solved and static.** They are a scarce, autoscaling, security-critical resource.
8. **Benchmark-only rewards.** Hard verifiers cover math and code; they do not cover instruction-following or general agent quality. A reward curriculum moving hard → soft is usually necessary.
9. **Budgeting on GPU-hour rate alone.** Egress, storage, and idle time dominate small budgets; spot instances get reclaimed, so checkpoint or lose the run.

---

## 11. Minimal reading list

| Paper | arXiv | Why |
|---|---|---|
| HybridFlow (verl) | [2409.19256](https://arxiv.org/abs/2409.19256) | The hybrid-controller architecture behind the ecosystem default |
| StreamRL | [2504.15930](https://arxiv.org/abs/2504.15930) | Colocated vs disaggregated; resource coupling; the bubbles |
| SeamlessFlow | [2508.11553](https://arxiv.org/abs/2508.11553) | Trainer/agent isolation; bubble-free pipelines |
| DeepSeekMath | [2402.03300](https://arxiv.org/abs/2402.03300) | **GRPO's original paper** — cite this, not R1, for the algorithm |
| DAPO | [2503.14476](https://arxiv.org/abs/2503.14476) | Open RL system at scale, fully reproducible |
| DeepSeek-R1 | [2501.12948](https://arxiv.org/abs/2501.12948) | The long-CoT recipe; why rollouts got long |
| ScaleRL | [2510.13786](https://arxiv.org/abs/2510.13786) | RL compute scaling laws; off-policy and variance |
| SortedRL | [2603.23414](https://arxiv.org/abs/2603.23414) | Online length-aware scheduling for the long tail |

---

**Back to:** [README](../README.md) · [ROADMAP](../ROADMAP.md) · [STACK L2b](../STACK.md#l2b--rl-post-training-infrastructure) · [inference math](inference-math.md)
