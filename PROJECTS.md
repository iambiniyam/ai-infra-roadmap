# Projects

Seventeen buildable projects, ordered by difficulty. Each one maps to a stage in [ROADMAP.md](ROADMAP.md) and ends with **acceptance criteria**: if you can't check every box, the project isn't done.

> **Rules of the game:** every project ends in a *number* and a *public artifact* (repo, README, plot, or blog post). Numbers because infra is empirical. Public because that's what actually gets you hired.

---

## How to use this file

1. Pick a project at your current stage.
2. Read the deliverables *before* starting so you know what "done" means.
3. Struggle first, then search. Stop when the acceptance criteria are met, not when the code looks nice.
4. Write a one-page README with methodology, results, and what surprised you.

---

## Stage 1 · Foundations of one GPU

### P1 · GPU Report Card

**Skills:** Stage 0-1

**Build a reproducible benchmark that characterizes the machine you have.**

Deliverables:

- `bench/` with scripts measuring: HBM copy bandwidth, PCIe host↔device bandwidth, disk read bandwidth (cache vs `O_DIRECT`), and a matmul TFLOPS sweep across dtypes.
- A results table for *your* hardware.
- A one-paragraph roofline interpretation: where is this machine compute-bound and where is it bandwidth-bound?

**Acceptance criteria**

- [ ] Matmul TFLOPS within 15% of the datasheet for bf16 at large M=N=K.
- [ ] You measured HBM bandwidth and it's within 20% of spec.
- [ ] You know your crossover intensity (FLOP/byte) and stated it.
- [ ] README explains what each number means for a real workload.

**Proves:** you can measure instead of guess.

---

### P2 · Rebuild a GPT, Reproducibly

**Skills:** Stage 1

**Train a small GPT from scratch and make the run reproducible.**

Use [nanoGPT](https://github.com/karpathy/nanoGPT) or write ~500 lines yourself. Train on a small corpus (e.g. TinyStories, Shakespeare, or a domain corpus).

Deliverables:

- Config-driven training (`config.yaml`), seeded, logged (MLflow or W&B-free alternative like [Aim](https://github.com/aimhubio/aim)).
- Loss curves for 2-3 hyperparameter variants.
- A **tokens/sec + MFU** measurement in the README.
- A sample of generated text before/after training.

**Acceptance criteria**

- [ ] A stranger can reproduce your loss curve from the README in one command.
- [ ] You computed MFU by hand and it matches the logged value.
- [ ] You can explain what changed between variants and why.
- [ ] You can state the model's memory breakdown (params/grads/optimizer/activations).

**Proves:** you understand training, not just `Trainer.train()`.

---

### P3 · Fine-Tune with an Eval Gate

**Skills:** Stage 1

**Fine-tune a 7B model on a real task and prove it got better, or admit it didn't.**

Use [Unsloth](https://github.com/unslothai/unsloth) or [LlamaFactory](https://github.com/hiyouga/LlamaFactory) with QLoRA. Build a 50-200 example eval set *first*.

Deliverables:

- LoRA/QLoRA run with config, VRAM, and time-to-train.
- Baseline vs fine-tuned scores on your eval set, plus an LLM-as-judge or exact-match rubric.
- A short write-up: what helped, what didn't, and the failure cases.

**Acceptance criteria**

- [ ] The eval set existed *before* training.
- [ ] You report both a win and a regression (there is always one).
- [ ] You measured the quality/cost tradeoff of at least two ranks or quant levels.
- [ ] You can explain why LoRA ≠ full fine-tuning and when it matters.

**Proves:** you evaluate instead of vibing.

---

### P4 · Local Inference Lab

**Skills:** Stage 1

**Serve a model locally and characterize the context-length cliff.**

Use [llama.cpp](https://github.com/ggml-org/llama.cpp) or [Ollama](https://github.com/ollama/ollama) with GGUF quants at 4k/16k/32k context.

Deliverables:

- tok/s and VRAM table across context lengths × quant levels.
- Predicted-vs-actual KV cache table (use the [KV formula](cheatsheets/inference-math.md)).
- A plot showing where throughput falls off.

**Acceptance criteria**

- [ ] Prediction error on VRAM < 15%.
- [ ] You explain the throughput decay with the bandwidth equation.
- [ ] You pick a quant for a stated constraint (VRAM or quality) and defend it.

**Proves:** you reason about memory before you run out of it.

---

### P5 · Kernel Autopsy

**Skills:** Stage 1-2

**Profile a real training or inference step and make it measurably faster.**

Deliverables:

- `nsys`/`torch.profiler` trace + the top-10 kernel time table.
- A change (fusion, dtype, `torch.compile`, flash-attention, better data loader) with before/after numbers.
- A roofline placement of the dominant kernel.

**Acceptance criteria**

- [ ] ≥ 15% end-to-end speedup (or a documented, evidence-backed dead end).
- [ ] You identified the dominant kernel and whether it's compute- or bandwidth-bound.
- [ ] You changed **one** variable and attributed the delta.

**Proves:** you optimize with evidence.

---

## Stage 2 · Multi-GPU & serving

### P6 · Latency-Throughput Lab

**Skills:** Stage 2

**Produce the canonical serving artifact: a latency-throughput curve.**

Serve a 7B-13B model on [vLLM](https://github.com/vllm-project/vllm). Sweep concurrency 1 → 512 with a fixed prompt distribution.

Deliverables:

- TTFT vs throughput plot, and TPOT/ITL vs throughput plot, with p50/p95/p99.
- The "knee" identified, plus the config that maximizes goodput at a stated SLO.
- Reproducible benchmark config (model, quant, `max_num_seqs`, `max_model_len`, GPU, driver).

**Acceptance criteria**

- [ ] Warmup excluded; steady-state measured.
- [ ] You define goodput explicitly and use it to pick a config.
- [ ] You can explain *why* TTFT and ITL move in opposite directions.
- [ ] Configs and versions are pinned in the README.

**Proves:** you can produce the number that product teams actually need.

---

### P7 · Cost-per-Token Optimizer

**Skills:** Stage 2

**Cut $/1M tokens without violating an SLO.**

Start from P6. Add, one at a time: FP8 KV cache, prefix caching, chunked prefill, `max_num_seqs` tuning, quantization (AWQ/FP8), speculative decoding.

Deliverables:

- A table: change → Δ TTFT → Δ throughput → Δ $/1M tokens → Δ quality.
- A final config with the best $/1M tokens at your SLO.

**Acceptance criteria**

- [ ] ≥ 2x improvement in $/1M tokens *or* documented evidence that the baseline was already optimal.
- [ ] Every change has its own measured delta.
- [ ] Quality was checked (not assumed) for any quantization change.
- [ ] You state the assumptions in the cost model (GPU $/hr, utilization).

**Proves:** you optimize the thing the business cares about.

---

### P8 · Multi-LoRA Serving

**Skills:** Stage 2

**Serve five adapters on one base model and measure the economics.**

Deliverables:

- One endpoint serving ≥5 LoRA adapters (train tiny ones if needed).
- Peak VRAM vs N adapters, and per-adapter latency.
- Comparison: multi-LoRA vs five separate deployments (memory and $).

**Acceptance criteria**

- [ ] Adapter routing is correct (no cross-contamination).
- [ ] You quantify the per-adapter memory overhead.
- [ ] You state when multi-LoRA is the wrong choice.

**Proves:** you can multiply serving capacity without multiplying GPUs.

---

### P9 · DDP → FSDP Scaling Study

**Skills:** Stage 2

**Measure scaling efficiency and explain it with the fabric.**

Deliverables:

- Tokens/sec and MFU for DDP(8×1), FSDP(2×4), FSDP(1×8) on the same job.
- Scaling efficiency curve and a comparison with measured interconnect bandwidth.
- A recommendation: which config for which model size, and why.

**Acceptance criteria**

- [ ] Numbers are reproducible (seeds, versions, node specs).
- [ ] You connect the efficiency loss to a specific collective and its cost.
- [ ] You state the model size at which your recommendation flips.

**Proves:** you understand sharding as a trade, not a checkbox.

---

## Stage 3 · Multi-node

### P10 · Fabric Report

**Skills:** Stage 3

**Characterize the network your training depends on.**

Run [nccl-tests](https://github.com/NVIDIA/nccl-tests) `all_reduce_perf` and `all_to_all` across 2+ nodes. Compare IB vs Ethernet if available.

Deliverables:

- Bus-bandwidth vs message-size plots for all-reduce, all-gather, and all-to-all.
- Comparison against theoretical fabric bandwidth.
- A paragraph on what model parallelism this fabric supports.

**Acceptance criteria**

- [ ] You ran a warmup and report steady-state bus bandwidth.
- [ ] You explain the small-message latency floor.
- [ ] You can say whether this fabric supports full MoE all-to-all, and at what scale.

**Proves:** you can predict cluster performance from a wire measurement.

---

### P11 · Fault-Injection Drill

**Skills:** Stage 3

**Break a multi-node job on purpose and recover it.**

Deliverables:

- A multi-node training job with periodic sharded checkpoints.
- A script that kills a node/worker at a random step.
- A recovery log with measured detection time, data loss, and RTO.

**Acceptance criteria**

- [ ] Recovery completes without manual surgery (or you document the surgery).
- [ ] RTO and RPO are numbers, not adjectives.
- [ ] You document the failure mode you didn't expect.
- [ ] Checkpoint write time is measured against your recovery objective.

**Proves:** you design for failure, not around it.

---

### P12 · Disaggregated Inference

**Skills:** Stage 3

**Deploy prefill/decode separation and find where it *hurts*.**

Use [llm-d](https://github.com/llm-d/llm-d) or [Dynamo](https://github.com/ai-dynamo/dynamo).

Deliverables:

- Colocated vs disaggregated results across prompt-heavy and decode-heavy workloads.
- TTFT/ITL/throughput comparison at equal GPU count.
- An explicit statement of the crossover (when disaggregation wins and when it loses).

**Acceptance criteria**

- [ ] Equal-hardware comparison (same GPU count).
- [ ] You show a case where disaggregation is *worse*.
- [ ] You explain the result with KV-transfer and scheduling costs.

**Proves:** you know when the fashionable architecture is wrong.

---

## Stage 4 · Platform

### P13 · Mini GPU Platform

**Skills:** Stage 4

**Stand up a shared GPU platform with quotas, telemetry, and isolation.**

Components: K8s + GPU Operator + Kueue/Volcano quotas + Prometheus + Grafana + DCGM exporter + a notebook/SSH path for users.

Deliverables:

- Architecture diagram and install docs (someone else can rebuild it).
- Per-namespace GPU utilization dashboard.
- A fairness demo: tenant A cannot starve tenant B.
- Idle-waste number in dollars per week.

**Acceptance criteria**

- [ ] A new team gets a working GPU environment in < 1 day using only your docs.
- [ ] The fairness demo is reproducible with a script.
- [ ] You can name the three biggest sources of waste and their cost.

**Proves:** you can run infrastructure other people depend on.

---

### P14 · FinOps Dashboard That Changes a Decision

**Skills:** Stage 4

**Make cost visible, then act on it.**

Deliverables:

- $/GPU-hour by team, $/1M tokens per endpoint, utilization vs allocation, and idle cost.
- One decision you made because of the dashboard (right-size, kill, migrate to spot, etc.).
- Before/after cost for that decision.

**Acceptance criteria**

- [ ] Numbers are reconciled with the actual cloud bill or internal chargeback.
- [ ] At least one decision changed.
- [ ] You documented the assumptions and their sensitivity.

**Proves:** you can talk to finance without flinching.

---

### P15 · Eval + Red-Team Pipeline in CI

**Skills:** Stage 4-5

**Make quality and safety a gate, not a hope.**

Deliverables:

- CI pipeline: eval suite ([lighteval](https://github.com/huggingface/lighteval)/[DeepEval](https://github.com/confident-ai/deepeval)/[Ragas](https://github.com/vibrantlabsai/ragas)) + red-team pass ([garak](https://github.com/NVIDIA/garak), [promptfoo](https://github.com/promptfoo/promptfoo)).
- A threshold that blocks a bad model, demonstrated.
- A regression alert wired to your tracker.

**Acceptance criteria**

- [ ] The gate blocked at least one model that looked fine by eyeball.
- [ ] Every eval has a fixed dataset version.
- [ ] Results are stored and diffable across model versions.

**Proves:** you can ship AI without shipping regressions.

---

## Stage 5 · Frontier

### P16 · Write a Triton Kernel

**Skills:** Stage 5

**Beat the PyTorch baseline on a real op in your workload.**

Deliverables:

- Kernel implementation with correctness tests vs a reference.
- Speedup and a roofline placement.
- A write-up of the optimization journey (tiling, masking, vectorization).

**Acceptance criteria**

- [ ] Numerically correct within tolerance across shapes and dtypes.
- [ ] Faster than the baseline on the shapes you care about (with a chart).
- [ ] You can explain the kernel's bottleneck without the profiler.

**Proves:** you can go below the framework.

---

### P17 · Reproduce a Paper

**Skills:** Stage 5

**Pick a systems/ML-systems paper and reproduce its headline claim.**

Good candidates: FlashAttention, PagedAttention, SARATHI, DistServe, AWQ, Ring Attention, a scheduling paper from MLSys/OSDI.

Deliverables:

- Reproduction code + results vs the paper's claim.
- An honest section on what didn't replicate and your best explanation.
- A public write-up.

**Acceptance criteria**

- [ ] You state the exact claim being tested and the evaluation config.
- [ ] You report a gap, not a copy of the abstract.
- [ ] You identify at least one assumption in the paper that your environment violated.

**Proves:** you can stand on the frontier instead of following it.

---

## Project README template

Copy this for every project:

```markdown
# <Project name>

## Question
What am I trying to learn or prove? (one sentence, falsifiable)

## Setup
Hardware / versions / model / dataset / configs (pinned).

## Method
What I measured, how, with what warmup and repetitions.

## Results
The table and/or plot. Raw numbers, not adjectives.

## Interpretation
Why these numbers. What bound was hit. What I got wrong.

## Reproduction
Exact commands.

## Next
What I'd do with 10x the budget.
```

---

**Suggested order for a portfolio:** P1 → P2 → P4 → P5 → P6 → P7 → P10 → P13 → P15. That sequence touches all nine layers and produces artifacts a hiring manager can evaluate in five minutes.
