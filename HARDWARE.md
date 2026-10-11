# Hardware, Math & Economics

Infra people are distinguished by one habit: they can predict the number before running the benchmark. This file gives you the tables and formulas to do that.

> **Caveat:** vendor numbers move every generation and marketing FLOPs assume dense, ideal conditions. Treat the tables as orders of magnitude and *verify against the datasheet* before you sign a PO. The **formulas** are the durable part.

> **Read every hardware number through this checklist.** Almost all hardware arguments in real meetings are actually disagreements about which of these six axes each person is quoting. Ask them explicitly:
>
> 1. **Which format?** bf16, fp8, fp4, mxfp4 — they differ by up to 4x on the same chip.
> 2. **Dense or sparse?** Vendors quote sparse when it flatters. Sparsity needs a model that exploits it.
> 3. **Per chip or per system?** "GB200 has 13.4 TB" means the 72-GPU rack, not one GPU.
> 4. **Spec or usable?** Datasheet capacity is not addressable capacity (192 vs 186 GB, 288 vs 270 GB).
> 5. **One-way or bidirectional?** PCIe, NVLink, and IB are all quoted both ways by different vendors.
> 6. **Peak or sustained?** Thermal throttling and power caps are real; a 2 kW part in a 1.4 kW slot runs slower.

---

## 1. The memory hierarchy (memorize the shape)

Every performance problem is a data-movement problem. These are approximate 2026 figures for a high-end GPU node:

| Tier | Capacity | Bandwidth | Latency | What it implies |
|---|---|---|---|---|
| Registers | ~256 KB/SM | n/a | ~1 cycle | Hand-tuned kernels live here |
| SRAM / L2 | 40-60 MB (L2) | ~10 TB/s | ~200 cycles | FlashAttention's whole trick is keeping tiles here |
| HBM (GPU) | 80-288 GB | 3.0-22 TB/s | ~600 cycles | Bandwidth sets decode speed |
| NVLink (intra-node) | n/a | 0.9-3.6 TB/s/GPU | n/a | The only "fast" place to do tensor parallelism |
| PCIe Gen5 | n/a | 63 GB/s (x16, one way) | n/a | ~14x slower than NVLink 4. Avoid TP over PCIe. |
| InfiniBand NDR/XDR | n/a | 50/100 GB/s per port | ~1-2 µs | Inter-node collectives. The real cluster currency. |
| Ethernet (RoCE 400G) | n/a | ~50 GB/s | ~2-5 µs | Cheaper fabric; tune it or you'll notice |
| Local NVMe | 4-30 TB | 3-14 GB/s | ~100 µs | Checkpoint staging |
| Shared parallel FS | PB scale | 100 GB/s to 1 TB/s aggregate | ~ms | Where checkpoints and datasets live |
| Object store (S3) | unlimited | ~GB/s per client | ~10s of ms | Durable, cheap, never on the hot path |

**PCIe nuance worth knowing:** PCIe 5.0 x16 encodes 32 GT/s × 16 lanes ÷ 8 = 64 GB/s, but 128b/130b line encoding makes the *effective* one-direction rate **63 GB/s** (126 GB/s bidirectional). Folklore quotes both. Know which one you're using: 63 is what a `cudaMemcpy` benchmark will actually approach.

**The one ratio to remember:** NVLink (≈900 GB/s) ÷ InfiniBand NDR (≈50 GB/s) ≈ **18x**. That ratio is why tensor parallelism stays inside a node and why pipeline/expert parallelism exist for crossing nodes.

> **But the 18x is not a law — it is a consequence of matched generations.** Scale-up (NVLink) bandwidth doubles *every* generation; scale-out (InfiniBand) per-port bandwidth doubles every *two*. So the ratio itself doubles each generation:
>
> | Pairing | Ratio |
> |---|---|
> | NVLink 4 (900 GB/s) ÷ IB NDR (50 GB/s) | 18x |
> | NVLink 4 ÷ IB XDR (100 GB/s) | 9x |
> | NVLink 5 (1800 GB/s) ÷ IB XDR | 18x |
> | NVLink 6 (3600 GB/s) ÷ IB XDR | **36x** |
>
> This is the actual argument for rack-scale domains: as the ratio grows, the node boundary becomes more punishing, so vendors widen the "node" (NVLink domains of 72, then 144+ GPUs). Memorize the *mechanism*, not the constant.

---

## 2. Accelerators (approximate, dense, verify)

| Chip | Memory | HBM BW | Dense compute | Notable |
|---|---|---|---|---|
| NVIDIA H100 SXM | 80 GB HBM3 | 3.35 TB/s | ~990 TF bf16 | The 2023-2025 workhorse for training and serving |
| NVIDIA H200 | 141 GB HBM3e | 4.8 TB/s | ~990 TF bf16 | **Same compute die as H100 — only memory changed.** That is why H200 wins on long context and loses nothing |
| NVIDIA B200 | 192 GB (186 usable) HBM3e | ~8 TB/s | ~2.25 PF bf16 / ~9 PF FP4 | FP4/FP8 focus; the format matters more than the number |
| NVIDIA GB200 NVL72 | 72 GPUs, 13.4 TB usable HBM | ~8 TB/s/GPU | ~2.5 PF bf16/GPU | Rack-scale NVLink domain; the rack *is* the node |
| NVIDIA B300 / GB300 | 288 GB (270-279 usable) HBM3e | ~8 TB/s | ~1.1x B200 | Blackwell Ultra. **Spec capacity ≠ usable capacity** |
| NVIDIA Rubin / VR NVL72 | 288 GB HBM4 | ~22 TB/s | ~50 PF NVFP4 | NVLink 6 at 3.6 TB/s; ~2 kW/chip; **first 100% liquid-cooled arch** |
| NVIDIA L40S | 48 GB GDDR6 | 0.86 TB/s | ~360 TF fp16 | Cheap inference/fine-tuning; **no NVLink**, so no TP |
| NVIDIA RTX PRO 6000 | 96 GB GDDR7 | ~1.8 TB/s | — | Workstation/edge inference with big memory |
| AMD MI300X / MI325X | 192 / 256 GB HBM3 | 5.3 / 6.0 TB/s | ~2.6 PF fp16 | Best $/GB memory for inference-heavy work |
| AMD MI350X / MI355X | 288 GB HBM3e | 8 TB/s | — | CDNA 4; the credible non-NVIDIA large-memory option |
| AMD MI400-series | 432 GB HBM4 | ~23 TB/s | ~40 PF MXFP4 | UALoE 72-GPU scale-up domain; open-standard answer to NVL72 |
| Google TPU v5e / v5p | 16 / 95 GB | 1.6 / 2.76 TB/s | ~200 / ~460 TF bf16 | Cost-efficient at scale if you're XLA-native |
| Google TPU v6e (Trillium) | 32 GB | — | ~900 TF | Previous-gen TPU |
| Google TPU v7 (Ironwood) | 192 GB HBM3e | — | ~4.6 PF FP8 | 9,216-chip superpod; designed for inference-era scale |
| AWS Trainium2 | ~96 GB | — | FP8-focused | Cheapest $/token for some workloads; op coverage is the catch |
| AWS Trainium3 | 144 GB HBM3e | 4.9 TB/s | ~2.5 PF FP8 | AWS has converged on Trainium; **no Inferentia 3** |
| Huawei Ascend 950PR/DT | 128 / 144 GB | up to 4 TB/s | — | Relevant in China; self-developed HBM; `vllm-ascend`/CANN |
| Apple M-series | Unified | ~0.1-0.5 TB/s | n/a | MLX/ANE; great local dev, not a cluster |

**How to pick:** training at scale → high HBM bandwidth *and* fast interconnect. Serving with long context → HBM capacity and bandwidth (H200, MI300X/MI350X). Cheap high-volume serving → the best $/token *after* accounting for ops cost (L40S, Trainium).

**Three traps in every spec table, including this one:**

1. **Format before FLOPs.** "2.2 PFLOPS" is meaningless without bf16/fp8/fp4 and dense/sparse. Prefer dense numbers; vendors quote sparse when it sounds better. FP4 is often ~4x the FP16 number on Blackwell-class parts — that is a format change, not a free speedup.
2. **Spec capacity ≠ usable capacity.** A B200 is "192 GB" but ~186 GB is addressable; a B300 is "288 GB" but an HGX B300 exposes ~270 GB and a GB300 NVL72 ~279 GB. Datasheets and capacity planning use different numbers — know which you need.
3. **"Supports" ≠ "accelerates."** A chip can *run* MXFP4 at MXFP8 speed. Ask what the silicon natively accelerates.

---

## 3. Precision & bytes per parameter

| Format | Bytes | Use |
|---|---|---|
| FP32 | 4.0 | Legacy; optimizer master weights |
| TF32 | 4.0 (19-bit mantissa) | Default matmul on Ampere+; free speedup |
| BF16 | 2.0 | **Training and serving default.** Wide range, good stability |
| FP16 | 2.0 | Slightly more precision, needs loss scaling |
| FP8 (E4M3/E5M2) | 1.0 | Modern training and serving; big bandwidth win, care with outliers |
| FP4 / NVFP4 | 0.5 | Blackwell-era serving; needs calibration |
| INT8 | 1.0 | Quantized inference; smooth via SmoothQuant-ish methods |
| INT4 / Q4 | 0.5 | GGUF/AWQ/GPTQ local serving |
| 1-bit / 1.58-bit | ~0.13-0.2 | Research (BitNet); surprising but check quality |

**Rule of thumb:** model size in GB ≈ params × bytes-per-param. A 70B model is ~140 GB in bf16, ~70 GB in fp8, ~35 GB in int4.

---

## 4. Memory math (the most useful section in this repo)

### 4.1 Inference memory

```text
weights              = params × bytes_per_param
KV cache             = 2 × n_layers × n_kv_heads × head_dim × seq_len × batch × kv_bytes
activations          = small relative to KV for decode; can be large for prefill of long sequences
overhead             = CUDA context, workspace, fragmentation  → budget 1-3 GB
----------------------------------------------------------------------------
total ≈ weights + KV cache + overhead + (prefill activations)
```

**Worked example: Llama-3-70B (80 layers, 8 KV heads, head_dim 128), bf16:**

| Quantity | Value |
|---|---|
| Weights (bf16) | 140 GB → **does not fit on one 80 GB GPU** |
| Weights (fp8) | 70 GB → fits, ~10 GB left for KV |
| Weights (int4) | ~35 GB → comfortable |
| KV per token per sequence | `2×80×8×128×2 = 327,680 B ≈ 0.31 MB` |
| KV at 8k context, 1 seq | ~2.6 GB |
| KV at 8k context, 32 seqs | ~84 GB |

**Conclusion you can derive, not guess:** 70B bf16 needs ≥2 GPUs with tensor parallelism; long context at high concurrency forces KV quantization, paged/prefix caching, or disaggregation. This is *the* reasoning that separates designers from configurers.

### 4.2 Training memory

```text
Mixed-precision Adam (typical) per parameter:
  bf16 weights          2 bytes
  fp32 master weights   4 bytes
  gradients             2-4 bytes
  Adam m + v            8 bytes
  -------------------------------------
  ≈ 16-20 bytes/parameter   →  a 7B model needs ~112-140 GB before activations

Plus activations, which scale with batch × sequence × layers × hidden × precision
(this is what activation checkpointing trades compute for).
```

**That's why you need sharding.** ZeRO-1/2/3 or FSDP partitions optimizer states, gradients, and parameters respectively across ranks. A 7B model trains on 8×80 GB because ~140 GB of state is split ~17 GB/GPU.

### 4.3 ZeRO stage cheat

| Stage | Shards | Memory saved | Communication |
|---|---|---|---|
| DDP | nothing | 0 | all-reduce gradients |
| ZeRO-1 | optimizer states | ~8 bytes/param | all-gather + reduce-scatter |
| ZeRO-2 | + gradients | +2-4 bytes/param | + reduce-scatter gradients |
| ZeRO-3 / FSDP | + parameters | up to full model | + all-gather params per layer |

---

## 5. Compute math

### 5.1 Training FLOPs

```text
C_train ≈ 6 × N × D
  N = parameters (non-embedding), D = training tokens

Why 6: 2 (forward) + 4 (backward) multiply-accumulates per param per token.
```

**Cost example: train an 8B model on 1T tokens:**

```text
C = 6 × 8e9 × 1e12 = 4.8e22 FLOPs
With bf16 peak 990 TFLOPS and a realistic 40% MFU → 3.96e14 FLOP/s effective
Time = 4.8e22 / 3.96e14 ≈ 1.21e8 s ≈ 33,600 GPU-hours
At $2.50/GPU-hour → ≈ $84,000
```

Now vary it: 70% MFU instead of 40% → ~$48k. **MFU is a money number.**

### 5.2 Inference FLOPs

```text
Prefill (compute-bound):  ≈ 2 × N × tokens_in   + attention term
Decode  (memory-bound):   ≈ 2 × N per token     <- but you're waiting on HBM, not FLOPs
```

### 5.3 The decode bandwidth limit (the single best back-of-envelope in serving)

```text
tokens/sec (batch 1) ≤ HBM_bandwidth / weights_bytes

H100: 3.35 TB/s ÷ 16 GB (8B model, bf16) ≈ 209 tok/s theoretical ceiling
                                  -> realistic single-stream: 60-120 tok/s
```

This is why quantization speeds up decode almost linearly with size reduction, and why batching is the only way to use a GPU's FLOPs efficiently: with batch 32, the same weight read serves 32 tokens.

---

## 6. Roofline & arithmetic intensity

```text
arithmetic intensity = FLOPs / bytes moved

If intensity < (peak FLOPs ÷ peak bandwidth) → memory-bound
If intensity > crossover                     → compute-bound

H100 crossover: 990e12 / 3.35e12 ≈ 295 FLOP/byte
```

| Workload | Intensity | Bound by |
|---|---|---|
| Decode, batch 1 | ~1-2 FLOP/byte | HBM bandwidth |
| Large GEMM | hundreds | Compute |
| Flash attention (fused) | moderate | SRAM bandwidth + compute balance |
| Elementwise ops | << 1 | Bandwidth (fusion is the fix) |

**Practical consequence:** if you're bandwidth-bound, make data smaller (quantize/fuse). If you're compute-bound, use a faster math mode (fp8) or a better algorithm.

---

## 7. Scaling & communication math

```text
All-reduce bytes per rank ≈ 2 × (N-1)/N × S      (S = payload size, N = ranks)
```

Consequences:

- **Data parallel** cost grows with model size (you all-reduce gradients) and is cheap per-step but scales badly past a point.
- **Tensor parallel** requires an all-reduce *per layer*, which is only viable at NVLink speeds, i.e. inside a node.
- **Pipeline parallel** trades idle "bubbles" for less communication; needs enough micro-batches to fill the pipeline.
- **Expert parallel (MoE)** needs all-to-all, the most demanding pattern; it changes your fabric requirements.

**Amdahl's law is the reason for "minimum parallelism that fits":** every added axis adds communication overhead somewhere, so scaling efficiency decays. Measure it (`nccl-tests`, MFU at 1 vs 8 vs 64 GPUs) rather than assuming it.

---

## 8. Cluster design

### Typical node (2026, 8-GPU)

| Component | Typical |
|---|---|
| GPUs | 8× H100/H200/B200 with NVSwitch (900 GB/s to 1.8 TB/s each) |
| CPUs | 2× 64-core (Grace/EPYC/Xeon) |
| RAM | 1-2 TB |
| NICs | 8× 400G IB (one per GPU, GPUDirect RDMA) |
| Local NVMe | 4-30 TB |
| Power | ~5.6 kW GPUs + 1.5 kW rest ≈ 7-10 kW/node |

### Sizing rules

| Resource | Rule of thumb |
|---|---|
| Fabric | Non-blocking or rail-optimized for training; some oversubscription is OK for inference |
| Checkpoint storage | Must write your full checkpoint in < 2 minutes; otherwise recovery dominates |
| Dataset throughput | ≥ 1 GB/s per GPU sustained, or the GPUs starve |
| Power | 8×H100 nodes → liquid cooling at rack density; plan for 40-80 kW/rack |
| Failure domains | Assume a GPU failure every few hours at 1,000+ GPUs; design for restart |

### Power and density (the constraint that replaced silicon)

Per-accelerator TDP has roughly **tripled in four years**: H100 ~700 W → B200 ~1,000 W → B300 ~1,400 W → Rubin ~2,000 W. Rack density went from <20 kW to **60-130 kW**, with exotic racks above 900 kW. At those densities **air cooling runs out of physical headroom** — this is a thermodynamics conclusion, not a vendor preference, and it is why every 2026 flagship rack is liquid-cooled.

| Metric | Air era | Liquid era |
|---|---|---|
| Rack density | 10-30 kW | 60-130 kW |
| PUE | 1.3-1.5 | 1.05-1.10 |
| Racks per MW | many, spread out | 50-70% fewer |
| Cooling share of build cost | modest | cold plates + CDUs dominate the mechanical budget |

**The reframing that matters:** you can buy GPUs; you often cannot buy megawatts. Datacenter power, not silicon supply, is now the binding constraint on most new clusters — which makes **tokens per second per megawatt** (and $/1M tokens/MW) the honest efficiency metric. FLOPS/W is a chip spec; tokens/s/MW is a business metric, and it is the one to bring to a capacity review.

### Failure arithmetic

```text
MTBF_cluster = MTBF_gpu / N

Empirical anchor: Meta's Llama-3 405B run used 16,384 H100s for 54 days and
logged 419 interruptions -- one every ~3 hours. That implies a per-GPU MTBF of
roughly 50,000 GPU-hours. Check it:

  16,384 GPUs -> 50,000 / 16,384  ≈ 3.1 h    (matches the observed ~3 h)
 100,000 GPUs -> 50,000 / 100,000 ≈ 0.5 h    (a failure every ~30 minutes)

Breakdown of those 419 interruptions: GPU faults 30.1%, HBM3 17.2%,
network 8.4%, CPU ~2%.  Note that ~47% is GPU+HBM, so "the GPU is fine" is
usually the wrong first assumption.
```

**How often should you checkpoint?** There is a closed form (Daly's model):

```text
optimal checkpoint interval = sqrt(2 × MTBF × checkpoint_write_time)
```

Worked at Llama-3 scale (~3 h MTBF): a 5-minute checkpoint write wants an
interval near **42 minutes**; a 10-minute write wants ~**60 minutes**. The
interval scales with the *square root* of write time, so making checkpoints 4x
cheaper only halves the interval. Pure checkpoint overhead at the optimum is
`2 × write_time / interval` — about 24% and 33% for those two cases, which is why
real systems overlap the write with compute (async/sharded checkpointing) rather
than pausing for it.

**The failure mode that costs more than crashes: fail-slow (stragglers).** A failing-but-not-failed GPU emits no XID error. Studies of production clusters find a majority of large jobs hit stragglers, producing tens of percent completion delay and low-single-digit-percent wasted GPU-hours — larger in aggregate than clean fail-stop losses. Detection requires *relative* signals (step-time outliers across ranks, `nccl-tests` re-runs, per-GPU throughput z-scores), not health flags. Practical consequences:

- Median utilization on real production fleets sits near **50%**, not 90%. Capacity plans built on 90% are fiction.
- Effective training time >90% is achievable *with* frequent checkpointing and automated retry; auto-retry recovers far more than manual triage.
- Checkpoint *restore* is often the slow step (tens of minutes), and it is frequently bottlenecked by metadata protocols (e.g. NFS RPC slot limits) rather than by network bandwidth.

That is why elastic training, frequent checkpoints, and node quarantine are not optional at scale.

---

## 9. Economics

### 9.1 The three cost metrics that matter

```text
$ / GPU-hour              → the unit price you negotiate
$ / 1M tokens (in/out)    → the price of what you actually sell
$ / 1M useful FLOPs       → MFU-adjusted efficiency (the honest one)
```

### 9.2 $/1M tokens, worked

```text
Measured: 1,500 output tok/s for a 2×H100 node at $5.00/node-hour
tokens/hour = 1,500 × 3,600 = 5.4M
$ / 1M tokens = 5.00 / 5.4 ≈ $0.93 per 1M output tokens
```

Now vary each lever and re-price: fp8 KV cache (+memory → +batch → +throughput), prefix caching (kills redundant prefill), speculative decoding (fewer sequential steps), disaggregation (better TTFT under long prompts), and quantization (less bandwidth per token). **This is the job.**

### 9.3 Buy vs rent

| Factor | Favors renting | Favors owning |
|---|---|---|
| Utilization | sporadic, < 40% | sustained, > 60-70% |
| Ops capacity | small team | dedicated platform team |
| Hardware generation risk | high (churn) | low (stable workload) |
| Data/compliance | cloud-friendly | must stay on-prem |
| Burst capacity | needed | predictable |

Break-even is usually **2-4x** in favor of ownership at high utilization, but only if you can staff the platform. Count the humans.

**The self-host-vs-API decision, stated honestly.** Self-hosting beats an API only when you clear the fixed costs:

| Situation | Usually wins |
|---|---|
| Serving a popular open base model at low volume | **The API.** You cannot beat someone else's utilization at 10% of yours |
| A fine-tuned or private model nobody hosts | **Self-hosting**, from day one — there is no alternative |
| High sustained utilization (roughly >70%) on hardware you already own | **Self-hosting**, and the margin grows with scale |
| Spiky or experimental demand | **The API**, unless latency/data residency forces otherwise |
| Latency-critical or data-resident workloads | **Self-hosting**, regardless of unit economics |

The reason is that an API provider amortizes across all their tenants' traffic and you amortize across yours. Your real levers are utilization, quantization, caching, and batching — not the hourly rate. **Cost per 1M tokens at a fixed SLO** is the metric; $/GPU-hour is just an input.

**Two 2026 pricing facts that surprise people** *(volatile — verify before you budget)*:

1. **Newest-generation silicon got *more* expensive per hour, not less.** Median on-demand rates for current-gen parts rose over 2024-2026 while previous-gen rates fell. Recency carries a premium; the $/hour curve is not monotonic in performance.
2. **The spread between providers for the same GPU is large** — commonly 2-4x between hyperscaler list price and neocloud spot. For the same nominal hardware, two independent price trackers can differ by ~2x. Never budget from a single quoted number.

---

## 10. Numbers worth memorizing

| Quantity | Value |
|---|---|
| Bytes per param: fp32/bf16/fp8/int4 | 4 / 2 / 1 / 0.5 |
| Training FLOPs per token per param | 6 |
| Inference FLOPs per token per param | 2 |
| H100 bf16 dense | ~990 TFLOPS (1,979 sparse — always say which) |
| H100 HBM bandwidth | ~3.35 TB/s |
| NVLink per-GPU bandwidth | ~0.9 TB/s (NVLink 4) / 1.8 (5) / 3.6 (6) |
| InfiniBand NDR / XDR per port | ~50 / ~100 GB/s |
| PCIe Gen5 x16 effective, one way | ~63 GB/s |
| H100 FP16/BF16 roofline crossover | ~295 FLOP/byte |
| Chinchilla compute-optimal tokens | ~20 × params |
| Practical tokens/param today | 100-1,000+ (way past Chinchilla) |
| Good MFU for large training | 40-55% |
| Median *production* GPU utilization | ~50% (plans built on 90% are fiction) |
| Per-GPU MTBF, well-run fleet | ~50,000 GPU-hours |
| Cluster MTBF | `MTBF_gpu / N` → 16k GPUs ≈ 3 h, 100k ≈ 30 min |
| Optimal checkpoint interval | `sqrt(2 × MTBF × write_time)` |

---

## 11. Procurement / capacity checklist

Before you buy or reserve anything, write down:

- [ ] **Workload mix**: training vs fine-tuning vs serving, and the ratio
- [ ] **Memory requirement**: largest model, context length, and concurrency
- [ ] **Interconnect requirement**: parallelism plan implies a fabric
- [ ] **Storage requirement**: dataset size, checkpoint size × frequency, bandwidth
- [ ] **Power and cooling**: kW/rack, and whether the building can take it
- [ ] **Utilization forecast**: honest, per quarter, with the idle cost priced in
- [ ] **Ops capacity**: who runs it at 3 a.m.?
- [ ] **Exit plan**: portability (vLLM/K8s/ONNX) so you're never locked in

**Next:** [cheatsheets/inference-math.md](cheatsheets/inference-math.md) for the serving formulas, [cheatsheets/training-parallelism.md](cheatsheets/training-parallelism.md) for the parallelism decision table.
