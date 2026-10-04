# The Roadmap

Six stages. Each one ends with **exit criteria**: concrete things you must be able to do before moving on. The stages compound: Stage 2 assumes you can profile; Stage 3 assumes you understand collectives.

> **How to read this file:** pick your stage, read only that section, build the matching project in [PROJECTS.md](PROJECTS.md), and come back when the exit criteria are true. Struggle first, search second, ask third.

**Jump to:** [Stage 0](#stage-0-foundations) · [Stage 1](#stage-1-single-gpu-fluency) · [Stage 2](#stage-2-single-node-multi-gpu-and-real-serving) · [Stage 3](#stage-3-multi-node-training-and-distributed-inference) · [Stage 4](#stage-4-platform-engineering) · [Stage 5](#stage-5-frontier-performance-and-scale) · [Cross-cutting skills](#cross-cutting-skills)

---

## Stage 0: Foundations

> **Goal:** be dangerous in a terminal, literate in the units of the field, and able to reason about bytes and bandwidth.
> **Prereq:** you can write code in some language.
> **Skip if:** you've operated Linux servers and know what a GPU's HBM bandwidth is.

The most common reason people bounce off AI infra is not AI; it's that they can't debug a container, read a profiler, or reason in bytes. Fix that first. It is a bounded amount of work, and everything above depends on it.

### Skills checklist

- [ ] **Linux:** processes, threads, `cgroups`/namespaces, virtual memory, page cache, `htop`/`iotop`/`nvtop`, `ssh`, `tmux`, `systemd`, `dmesg`
- [ ] **Python at infra level:** packaging (`uv`/`pip`), virtualenvs, type hints, `asyncio` basics, generators, and reading someone else's stack trace
- [ ] **Containers:** images vs layers, `Dockerfile`, volumes, user/UID issues, and GPU passthrough (`--gpus all`)
- [ ] **Kubernetes basics:** pod, deployment, job, service, configmap, `kubectl logs/exec/describe`, and *why* a pod is `Pending`
- [ ] **Networking:** IP/TCP, DNS, latency vs bandwidth, NICs, and the idea of RDMA/InfiniBand (not the details yet)
- [ ] **Storage:** POSIX vs object storage, the S3 API, throughput (GB/s) vs IOPS, and why many small files are poison for training
- [ ] **Units and precision:** bytes vs GB vs GiB, FLOPs vs FLOPS, TFLOPS vs effective TFLOPS, fp32/fp16/bf16/fp8/int8/int4, and what each costs in bytes
- [ ] **Python profiling:** `cProfile`, `py-spy`, and reading a flame graph

### The three numbers to internalize

| Quantity | Order of magnitude | Why it matters |
|---|---|---|
| HBM bandwidth (H100) | ~3.35 TB/s | Most LLM inference at batch 1 is *bandwidth*-bound, not compute-bound |
| NVLink (H100) | ~900 GB/s per GPU | Intra-node tensor parallelism lives or dies here |
| InfiniBand (NDR, per port) | ~50 GB/s | Inter-node all-reduce is ~18x slower than NVLink; this drives every multi-node design |

### Labs

1. **Bytes by hand.** For a 7B-parameter model, compute the size in fp32 / bf16 / fp8 / int4. Then compute the *training* memory including gradients and Adam states. Then the *serving* memory including KV cache. Verify against reality with `torch.cuda.memory_summary()`.
2. **Read a GPU.** Run `nvidia-smi -q`, `nvidia-smi dmon`, and `dcgmi dmon`. Identify power draw, memory used, SM utilization, and temperature under load.
3. **Containerize with a GPU.** Write a `Dockerfile` that runs a torch program with CUDA access, run it with `--gpus all`, and prove `torch.cuda.is_available()` inside.
4. **Measure bandwidth.** Write a benchmark that copies 1 GB through RAM, through disk (page cache vs `O_DIRECT`), and across PCIe to the GPU. Report GB/s for each. You now know the hierarchy.
5. **Find a bottleneck.** Run a deliberately slow data-loading loop; use `htop`/`iotop`/`py-spy` to identify whether CPU, disk, or GPU is the limit.

### Free resources

| Resource | Why |
|---|---|
| [MIT Missing Semester](https://missing.csail.mit.edu/) | Shell, git, debugging, profiling, the actual prereqs |
| [MIT 6.172 Performance Engineering](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/) | Makes you think in cache lines and cycles |
| [Kubernetes docs + "Kubernetes the Hard Way"](https://github.com/kelseyhightower/kubernetes-the-hard-way) | Understand K8s instead of cargo-culting it |
| [Brendan Gregg: Systems Performance](https://www.brendangregg.com/systems-performance-2nd-edition-book.html) | The USE method; free chapters online |
| [Latency Numbers Every Programmer Should Know](https://gist.github.com/jboner/2841832) | Memorize these |
| [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/) | Start watching now; it pays off in Stage 1 |

### Exit criteria

- You can state, without looking it up, why a 7B model needs ~14 GB of free VRAM in bf16 *before* activations.
- You can run a GPU container and show `nvidia-smi` output from inside it.
- You know your machine's memory bandwidth, disk bandwidth, and PCIe generation.
- You can look at a stuck process and say whether it's CPU-, memory-, IO-, or GPU-bound.

**Common trap:** collecting courses. Do the labs; skip anything that doesn't end in a number.

---

## Stage 1: Single-GPU fluency

> **Goal:** own one GPU end to end: train, fine-tune, serve, profile, and predict.
> **Prereq:** Stage 0.
> **Skip if:** you've profiled a training run and can compute a KV cache size.

### Skills checklist

- [ ] **PyTorch internals that matter:** autograd, `dtype`s, the caching allocator, `torch.cuda.memory_summary()`, `torch.compile`, `torch.amp`
- [ ] **Precision:** bf16 vs fp16 (and when loss scaling is required), fp8 on modern GPUs, and where numerical error actually shows up
- [ ] **Memory accounting:** params + gradients + optimizer states + activations + KV cache; activation checkpointing (recomputation) as a memory/speed trade
- [ ] **Profiling:** `torch.profiler`, Nsight Systems (`nsys`) for timeline, Nsight Compute (`ncu`) for a single kernel
- [ ] **The roofline model:** arithmetic intensity, compute-bound vs memory-bound, and why LLMs are usually memory-bound
- [ ] **MFU:** model FLOPs utilization = achieved / peak. Connect it to your training bill.
- [ ] **Fine-tuning:** full FT vs LoRA vs QLoRA, dataset formats, chat templates, packing, and *evaluation of the fine-tune*
- [ ] **Local inference:** llama.cpp/Ollama, GGUF quantization levels, and the context-length ↔ KV-cache ↔ VRAM relationship
- [ ] **The KV cache formula** (memorize it): `bytes ≈ 2 × layers × kv_heads × head_dim × seq_len × batch × dtype_bytes`

### Labs

1. **Train a tiny LLM.** Use `nanoGPT` or train GPT-2-small on a small corpus. Record tokens/sec, step time, and MFU. Compute MFU by hand and check your number.
2. **Fine-tune a 7B with QLoRA.** Measure peak VRAM and time. Then try LoRA with a larger rank and compare quality on a held-out set.
3. **Serve locally and measure.** Run a 7B at 4k, 16k, and 32k context in llama.cpp. Record tok/s and VRAM. Predict VRAM with the KV cache formula, then verify. Explain the gap.
4. **Profile and fix.** Profile a training step with `nsys` + `torch.profiler`. Find the dominant kernel. Make the step measurably faster (dtype, fusion, `torch.compile`, or flash-attention) and write down the delta.
5. **Quantize and evaluate.** Compare fp16 vs int8 vs Q4_K_M on (a) size, (b) tok/s, (c) quality via perplexity or an eval. Quantify the trade, don't vibe it.
6. **Roof a model.** For your GPU, compute the roofline crossover point (ops/byte) and decide whether your workload is compute- or memory-bound. Defend it.

### The formula that matters most

```text
KV cache bytes = 2 (K and V) × n_layers × n_kv_heads × head_dim × seq_len × batch × bytes_per_value

Example, Llama-3 8B (32 layers, 8 KV heads, head_dim 128, bf16):
  4k context, batch 1  → 2×32×8×128×4096×1×2  ≈ 0.54 GB
  32k context, batch 1 →                                ≈ 4.3 GB
  32k context, batch 32 →                               ≈ 138 GB   ← why long-context serving is hard
```

This one line explains GQA, prefix caching, paged attention, KV quantization, chunked prefill, and most of vLLM's design.

### Free resources

| Resource | Why |
|---|---|
| [Sebastian Raschka: LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) | Build a transformer from scratch, with no gaps |
| [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html) + [nanoGPT](https://github.com/karpathy/nanoGPT) | The clearest path from math to code |
| [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) | Transformers, tokenizers, fine-tuning, PEFT |
| [Unsloth notebooks](https://github.com/unslothai/unsloth) | Fastest way to a working fine-tune |
| [Horace He: Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) | The best intuition for compute- vs memory-bound |
| [PyTorch profiler recipes](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | Actually use the profiler |
| [GPU MODE lectures](https://github.com/gpu-mode/lectures) | CUDA/Triton/performance, straight from practitioners |
| [Modal / Together LLM fine-tuning guides](https://modal.com/docs/examples) | Practical recipes with cost math |

### Exit criteria

- Given a model, GPU, and batch size, you predict VRAM and throughput within 2x; then explain the error.
- You can explain why single-stream LLM inference is memory-bandwidth-bound and what changes at large batch.
- You compute a KV cache for any config in under a minute.
- You produce an MFU number for your training run and say whether it's good.
- You can state the quality/size/speed trade of at least three quantization schemes.

**Common trap:** fine-tuning before evaluation. If you can't measure the delta, you didn't fine-tune. You perturbed the weights and hoped.

---

## Stage 2: Single-node multi-GPU and real serving

> **Goal:** use all 8 GPUs in a box well, and serve a model to a latency/cost target.
> **Prereq:** Stage 1.
> **Skip if:** you've published a latency-throughput curve and a $/1M-token figure.

### Skills checklist

- [ ] **Collectives:** all-reduce, all-gather, reduce-scatter, all-to-all; ring vs tree; why all-to-all is the MoE tax
- [ ] **Topology:** NVLink/NVSwitch vs PCIe, NUMA affinity, and how topology dictates parallel strategy
- [ ] **Data parallel → sharded:** DDP → FSDP/ZeRO-1/2/3, and the memory-vs-communication trade at each stage
- [ ] **Parallelism basics:** tensor parallel (TP) and pipeline parallel (PP): enough to configure them, not just name them
- [ ] **Serving engine internals:** PagedAttention, continuous batching, chunked prefill, prefix caching, KV-cache blocks, scheduling
- [ ] **Serving metrics:** TTFT, TPOT/ITL, tokens/sec, **goodput**, and p50/p95/p99, plus how batch size moves each
- [ ] **Serving quantization:** FP8, AWQ, GPTQ, NVFP4, with an accuracy check
- [ ] **Advanced decoding:** speculative decoding, structured output/grammars, and multi-LoRA serving
- [ ] **Benchmarking discipline:** warmup, steady state, concurrency sweeps, fixed prompt distributions, reproducible configs

### Labs

1. **Scaling sweep.** Run the same training job as DDP (8×1) and FSDP (1×8). Report tokens/sec and scaling efficiency. Explain the gap using NCCL topology.
2. **Latency-throughput curve.** Serve a 7B-13B on vLLM. Sweep concurrency from 1 → 512. Plot TTFT vs throughput and identify the knee. Save the plot; it's your portfolio artifact.
3. **Beat your own baseline.** Starting from Lab 2, enable prefix caching, FP8 KV cache, chunked prefill, and `--max-num-seqs` tuning. Log each change with its delta in ms and $.
4. **Cost per token.** From measured throughput and the GPU's $/hour, compute $/1M input and output tokens. Optimize *cost per token at fixed SLO*, not raw tok/s.
5. **Multi-LoRA.** Serve one base model with 5 adapters. Measure memory overhead and per-adapter latency. Explain why this is cheaper than 5 models.
6. **Write an SLO.** Define p95 TTFT < 500 ms and p99 ITL < 50 ms for a chat product. Find the cheapest config that satisfies it. Write the one-page memo.

### Free resources

| Resource | Why |
|---|---|
| [vLLM docs](https://docs.vllm.ai/) + [vLLM paper (PagedAttention)](https://arxiv.org/abs/2309.06180) | Read the docs *and* the paper |
| [SGLang docs](https://docs.sglang.ai/) + [RadixAttention paper](https://arxiv.org/abs/2312.07104) | The other major engine; prefix-cache-first design |
| [Hugging Face: Efficient Training on Multiple GPUs](https://huggingface.co/docs/transformers/perf_train_gpu_many) | FSDP/TP/PP decision guide |
| [DeepSpeed ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/) | The canonical sharding explanation |
| [NVIDIA: Mastering LLM Techniques: Inference](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) | Batching, KV cache, quantization in one place |
| [Databricks: LLM Inference Performance Engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) | Real numbers and best practices |
| [Anyscale: Continuous batching & serving](https://www.anyscale.com/blog/continuous-batching-llm-inference) | Understand the scheduler |
| [vLLM `benchmarks/`](https://github.com/vllm-project/vllm/tree/main/benchmarks) | Steal their methodology |

### Exit criteria

- You have a latency-throughput curve for a real model and can pick a config for a stated SLO.
- You produce a $/1M-token number and defend the assumptions.
- You explain PagedAttention and continuous batching without notes.
- Given two GPUs with different interconnects, you predict scaling efficiency.
- You can read an `nsys` trace and point at the communication bottleneck.

**Common trap:** optimizing tok/s at batch 512 when your users are chatting at concurrency 5. Optimize for *your* workload's SLO, then look at cost.

---

## Stage 3: Multi-node training and distributed inference

> **Goal:** cross the node boundary: the place where most people's intuition breaks.
> **Prereq:** Stage 2, plus access to at least two nodes (cloud spot instances are fine).
> **Skip if:** you've debugged an NCCL hang across nodes to a specific rank and link.

### Skills checklist

- [ ] **Parallelism taxonomy:** DP, TP, PP, EP, SP, CP, FSDP/HSDP, and 2D/3D/4D composition, plus how to *choose* for a given model/cluster
- [ ] **Interconnects:** InfiniBand vs RoCE, fat-tree vs rail-optimized topology, NCCL algorithm selection, and NVLink vs IB bandwidth ratios
- [ ] **MoE specifics:** expert parallelism, all-to-all, capacity factors, load balancing, and communication cost
- [ ] **Checkpointing:** sharded, async, and resumable; storage bandwidth required to hit your recovery objective
- [ ] **Fault tolerance:** node/GPU failure, elastic training, stragglers, silent data corruption, NCCL timeouts, and flight recorder
- [ ] **Schedulers:** Slurm (`sbatch`/`srun`, `--gres`, topology-aware placement) **or** K8s (JobSet, LeaderWorkerSet, Kueue, Volcano, GPU Operator, DRA)
- [ ] **Disaggregated inference:** prefill/decode separation, KV-cache transfer (NIXL), routing, and when it actually helps
- [ ] **GPU health:** XID errors, ECC, `dcgmi diag`, burn-in tests, and how to quarantine a bad card

### Labs

1. **Measure the wire.** Run `nccl-tests` `all_reduce_perf` across 2 nodes. Compare IB vs Ethernet. Report bus bandwidth and the algorithmic crossover point.
2. **Compose parallelism.** Train with TP=2 × PP=2 × DP=2 and with pure DP on the same total GPU count. Compare MFU. Explain the delta using the interconnect numbers from Lab 1.
3. **Break it on purpose.** Run a multi-node job, kill a node mid-run, and resume from the last checkpoint. Write the failure story: detection time, data loss, recovery time.
4. **Go disaggregated.** Deploy prefill/decode separation (Dynamo or llm-d) with a long-prompt workload. Show the TTFT improvement vs a colocated deployment and the conditions where it *hurts*.
5. **Serve MoE.** Run an MoE model with expert parallelism. Measure all-to-all time as a fraction of step time and tune batch size.
6. **Health playbook.** Inject a bad GPU (or simulate with `CUDA_VISIBLE_DEVICES` + error injection). Detect it via DCGM/XID before it corrupts a run. Document the quarantine procedure.

### Free resources

| Resource | Why |
|---|---|
| [stas00/ml-engineering](https://github.com/stas00/ml-engineering) | The best free book on the debugging half of this stage |
| [Megatron-LM docs & paper](https://github.com/NVIDIA/Megatron-LM) | The reference for TP/PP/EP at scale |
| [NCCL tests & NVIDIA docs](https://github.com/NVIDIA/nccl-tests) | Measure before you theorize |
| [Google: How to Scale Your Model](https://jax-ml.github.io/scaling-book/) | Rigorous, hardware-grounded scaling math (JAX-flavored, universally useful) |
| [DeepSpeed / FSDP + TP docs](https://huggingface.co/docs/transformers/perf_train_gpu_many) | Practical configs |
| [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) + [llm-d](https://github.com/llm-d/llm-d) | Production disaggregated serving |
| [Slurm quickstart](https://slurm.schedmd.com/quickstart.html) | The HPC scheduler you'll meet at every lab |
| [Kueue](https://kueue.sigs.k8s.io/) + [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws) | K8s-native multi-node ML |

### Exit criteria

- Given a model, cluster, and budget, you write a parallelism plan and defend each axis.
- You read `nccl-tests` output and say whether the fabric is healthy.
- You can take an NCCL timeout and find the slow rank and the layer where it stalled.
- You explain why MoE changes network requirements and which topology you'd want.
- You can show a checkpoint/resume with a documented RTO and RPO.

**Common trap:** assuming "more parallelism = faster." Every added axis adds communication. The skill is finding the minimum parallelism that fits.

---

## Stage 4: Platform engineering

> **Goal:** run a shared GPU platform that other teams trust, and know what it costs.
> **Prereq:** Stage 3.
> **Skip if:** you've owned an SLO for a shared AI service and a FinOps dashboard that changed a decision.

At this stage the bottleneck stops being technical and becomes organizational: fairness, cost, reliability, and other people's deadlines.

### Skills checklist

- [ ] **Multi-tenancy:** quotas, fairness, gang scheduling, preemption, priorities, MIG partitioning, time-slicing, HAMi, and isolation guarantees
- [ ] **Compute supply:** reserved vs on-demand vs spot, autoscaling (Karpenter), gang scheduling for training, and capacity planning with headroom math
- [ ] **Storage architecture:** object store + parallel FS + caching tiers, the many-small-files problem, and checkpoint-write bandwidth
- [ ] **Data platform:** lakehouse formats (Iceberg/Delta/Lance/Hudi), batch vs streaming, feature stores, labeling workflows, dataset versioning and lineage
- [ ] **MLOps lifecycle:** experiment tracking → registry → lineage → reproducible pipelines → rollout/rollback, with CI/CD for models
- [ ] **Observability:** metrics (DCGM/Prometheus/Grafana), distributed traces (OTel → Langfuse/Phoenix), logs, and AI-specific SLOs
- [ ] **FinOps:** $/GPU-hour, utilization vs allocation, idle waste, right-sizing, cost per 1M tokens, chargeback/showback
- [ ] **Security & governance:** SBOM/Sigstore model supply chain, secrets, network policy, tenant isolation, provenance, egress control, audit
- [ ] **Reliability:** SLO/SLI, error budgets, DR, capacity headroom, incident response, and model-regression rollback

### Labs

1. **Build the mini-platform.** K8s + GPU Operator + Kueue quotas + Prometheus/Grafana with DCGM dashboards. Show per-namespace GPU utilization and quantify idle waste in dollars.
2. **Cost dashboard that changes a decision.** Build $/1M tokens per endpoint and $/GPU-hour by team. Use it to kill or right-size one workload. Write the before/after.
3. **Data lake for multimodal.** Object store + Iceberg or Lance + Ray Data preprocessing. Report throughput (GB/s) and $/TB processed.
4. **Ship with gates.** CI/CD: train → eval gate → registry → canary → automated rollback. Define the eval threshold that blocks a release.
5. **Prove isolation.** Demonstrate a runaway tenant cannot starve a neighbor (quotas + gang scheduling + preemption). Show the metrics.
6. **Write the SLOs.** Define SLIs/SLOs/error budgets and a capacity plan. Simulate a 30% capacity loss and show what breaks.

### Free resources

| Resource | Why |
|---|---|
| [Google SRE Book](https://sre.google/books/) | SLOs, error budgets, and incident management. Free |
| [The Datacenter as a Computer](https://research.google/pubs/the-datacenter-as-a-computer-an-introduction-to-the-design-of-warehouse-scale-machines/) | Warehouse-scale thinking, free PDF |
| [Kubernetes + Kueue + Volcano docs](https://kueue.sigs.k8s.io/) | The quota/fairness primitives |
| [Designing Machine Learning Systems: Chip Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) | The lifecycle; companion [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) is free |
| [MLFinOps / DCGM + Prometheus](https://github.com/NVIDIA/dcgm-exporter) | GPU telemetry you can chart |
| [FinOps Foundation](https://www.finops.org/framework/) | The vocabulary your finance team uses |
| [CNCF Landscape](https://landscape.cncf.io/) | Where the platform pieces come from |

### Exit criteria

- You defend a capacity plan with utilization, growth, and headroom numbers.
- You show a cost dashboard that changed a real decision.
- A new team can onboard onto your platform in a day using your docs.
- You can walk through recovery from a storage outage and a scheduler outage.
- Your eval gate has blocked at least one bad model.

**Common trap:** building a platform nobody asked for. Measure adoption and time-to-first-model, not features shipped.

---

## Stage 5: Frontier performance and scale

> **Goal:** be the person who makes the whole system faster, cheaper, and harder to break.
> **Prereq:** Stage 4, or deep specialization in one layer.
> **Skip if:** your changes already move production dashboards by 2x.

### Skills checklist

- [ ] **Kernels:** write a real Triton kernel; understand GEMM tiling, flash-attention, epilogues, warp specialization, and FP8/FP4 numerics
- [ ] **Compilers:** `torch.compile`, XLA, TVM, MLIR/IREE; graph capture, CUDA graphs, fusion, and layout choice
- [ ] **Serving architecture at scale:** disaggregation, KV-cache-aware routing, SLO-aware autoscaling, cascades, speculative decoding, cache-aware load balancing
- [ ] **Modern model shapes:** MoE, long context (ring/sequence parallel, sparse or linear attention), multimodal towers, and their infra implications
- [ ] **Hardware co-design:** read a chip datasheet (HBM, SRAM, TMA/TC, interconnect) and shape the algorithm to it
- [ ] **Benchmarking rigor:** MLPerf/InferenceMAX methodology, warmup, thermal effects, tail latency, statistical honesty
- [ ] **Research literacy:** read and reproduce MLsys/OSDI/NSDI/SOSP/SC/ICML results
- [ ] **Leverage:** design docs, incident reviews, mentoring, vendor strategy, and knowing when *not* to build

### Labs

1. **Write a kernel.** Implement a Triton kernel that beats the PyTorch baseline for a real op in your workload. Report the speedup and where it sits on the roofline.
2. **Reproduce a paper.** Pick a quant/attention/scheduling paper and reproduce its headline result end-to-end. Publish the delta and what didn't replicate.
3. **2x the economics.** Cut $/1M tokens by 2x at a fixed SLO without changing hardware. Write the design doc and the rollback plan.
4. **Red-team your own endpoint.** Run `garak` + `promptfoo` against a deployed service. Fix the top 3 findings and re-test.
5. **Design a big cluster.** Spec a 4,096-GPU (or largest defensible) cluster: topology, network, storage, power, failure domains, scheduler, and a cost model.

### Free resources

| Resource | Why |
|---|---|
| [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/) | Learn kernels for real |
| [How to Scale Your Model (Google)](https://jax-ml.github.io/scaling-book/) | Roofline-to-cluster scaling, free |
| [MLSys / OSDI / NSDI / SOSP / SC proceedings](https://www.mlsys.org/) | Where the next 5 years are published |
| [MLPerf & InferenceMAX results](https://mlcommons.org/benchmarks/) | The benchmark methodology standard |
| [SemiAnalysis](https://semianalysis.com/) + [GPU MODE](https://github.com/gpu-mode/lectures) | Hardware reality and kernel craft |
| [Efficient Deep Learning Systems (Yandex)](https://github.com/mryab/efficient-dl-systems) | Free course, systems-level |

### Exit criteria

- Your optimizations appear as numbers on a production dashboard.
- You read a hardware datasheet and predict performance before benchmarking.
- A design doc you wrote got adopted by another team.
- You can reproduce a frontier-lab result (at small scale) and explain the gap.

---

## Cross-cutting skills

These are not a stage. They're multipliers that start at Stage 2 and never stop.

| Skill | What "good" looks like | Start learning |
|---|---|---|
| **Measurement** | Every claim has a number, a config, and a date | Stage 1 |
| **Cost modeling** | You can price a workload before you build it | Stage 2 |
| **Reliability** | You design for failure, not around it | Stage 3 |
| **Security** | Supply chain, isolation, and prompt-injection are in your threat model | Stage 4 |
| **Writing** | A one-page design doc beats a two-hour meeting | Stage 1 |
| **Teaching** | You can explain PagedAttention to a smart backend engineer | Stage 2 |

---

## A suggested build order

If you are starting from scratch, do the work in this order. There is deliberately no schedule attached: how long each step takes depends on how much time you have and how much of it you already know.

| Order | Do this | Proof you did it |
|---|---|---|
| 1 | Stage 0 labs 1 to 4, plus one CS336 or GPU MODE lecture | A bytes-and-bandwidth memo with your machine's real numbers |
| 2 | Stage 1 labs 1 to 3 | A fine-tune with a loss curve, plus a tok/s table at three context lengths |
| 3 | Stage 1 labs 4 to 6, then Stage 2 labs 1 and 2 | A profile showing a real speedup, plus a latency-throughput plot |
| 4 | Stage 2 labs 3 to 6 | A benchmark report with $/1M tokens and an SLO memo |
| 5 | Stage 3 labs 1 and 2, using spot instances if you have no cluster | An `nccl-tests` report and a parallelism plan with an MFU comparison |
| 6 | Write it all up | One public blog post with real numbers |

Go at whatever pace you can sustain, and repeat any step that did not stick. The blog post at the end, with real numbers and honest failures, is worth more than any certificate, and it is the artifact that gets you interviews.

---

**Next:** [PROJECTS.md](PROJECTS.md) for the buildable version of every lab, [STACK.md](STACK.md) to choose tools, [cheatsheets/](cheatsheets/) for the formulas.
