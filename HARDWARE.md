# 🔩 Hardware, Math & Economics

Infra people are distinguished by one habit: they can predict the number before running the benchmark. This file gives you the tables and formulas to do that.

> **Caveat:** vendor numbers move every generation and marketing FLOPs assume dense, ideal conditions. Treat the tables as orders of magnitude and *verify against the datasheet* before you sign a PO. The **formulas** are the durable part.

---

## 1. The memory hierarchy (memorize the shape)

Every performance problem is a data-movement problem. These are approximate 2026 figures for a high-end GPU node:

| Tier | Capacity | Bandwidth | Latency | What it implies |
|---|---|---|---|---|
| Registers | ~256 KB/SM | — | ~1 cycle | Hand-tuned kernels live here |
| SRAM / L2 | 40–60 MB (L2) | ~10 TB/s | ~200 cycles | FlashAttention's whole trick is keeping tiles here |
| HBM (GPU) | 80–192 GB | 3.0–8.0 TB/s | ~600 cycles | Bandwidth sets decode speed |
| NVLink (intra-node) | — | 0.9–1.8 TB/s/GPU | — | The only "fast" place to do tensor parallelism |
| PCIe Gen5 | — | 64 GB/s (x16) | — | 25x slower than NVLink. Avoid TP over PCIe. |
| InfiniBand NDR/XDR | — | 50/100 GB/s per port | ~1–2 µs | Inter-node collectives. The real cluster currency. |
| Ethernet (RoCE 400G) | — | ~50 GB/s | ~2–5 µs | Cheaper fabric; tune it or you'll notice |
| Local NVMe | 4–30 TB | 3–14 GB/s | ~100 µs | Checkpoint staging |
| Shared parallel FS | PB scale | 100 GB/s–1 TB/s aggregate | ~ms | Where checkpoints and datasets live |
| Object store (S3) | ∞ | ~GB/s per client | ~10s of ms | Durable, cheap, never on the hot path |

**The one ratio to remember:** NVLink (≈900 GB/s) ÷ InfiniBand (≈50 GB/s) ≈ **18x**. That ratio is why tensor parallelism stays inside a node and why pipeline/expert parallelism exist for crossing nodes.

---

## 2. Accelerators (approximate, dense, verify)

| Chip | Memory | HBM BW | bf16 dense | Notable |
|---|---|---|---|---|
| NVIDIA H100 SXM | 80 GB HBM3 | 3.35 TB/s | ~990 TFLOPS | The 2023–2025 workhorse for training and serving |
| NVIDIA H200 | 141 GB HBM3e | 4.8 TB/s | ~990 TFLOPS | Same compute, way more memory & bandwidth → better for long-context serving |
| NVIDIA B200 | 192 GB HBM3e | ~8 TB/s | ~2.2 PFLOPS | FP4/FP8 focus; big step on memory bandwidth |
| NVIDIA GB200 NVL72 | 72 GPUs, 13.5 TB HBM | ~8 TB/s/GPU | — | Rack-scale domain; NVLink across 72 GPUs changes parallelism design |
| NVIDIA L40S | 48 GB GDDR6 | 0.86 TB/s | ~360 TFLOPS (fp16) | Cheap inference/fine-tuning; weak interconnect |
| AMD MI300X / MI325X | 192 GB / 256 GB HBM3 | ~5.3 TB/s | ~1.3 PFLOPS | Best $/GB memory for inference-heavy work |
| Google TPU v5e / v5p | 16 / 95 GB | — | ~200 / ~460 TFLOPS | Cost-efficient at scale if you're XLA-native |
| Google TPU v6e (Trillium) | 32 GB | — | ~900 TFLOPS | Current-gen TPU, strong perf/$ |
| AWS Trainium2 | ~96 GB | — | ~FP8-focused | Cheapest $/token for some workloads; op coverage is the catch |
| AWS Inferentia2 | 32 GB | — | — | Inference-only; good $ for standard models |
| Intel Gaudi 3 | 128 GB HBM2e | ~3.7 TB/s | ~1.8 PFLOPS (fp8) | Cost play; software maturity is the risk |
| Huawei Ascend 910B | 64 GB | — | — | Relevant in China; `vllm-ascend` and CANN ecosystem |
| Apple M-series | Unified | ~0.1–0.5 TB/s | — | MLX/ANE; great local dev, not a cluster |

**How to pick:** training at scale → high HBM bandwidth *and* fast interconnect (H100/H200/B200). Serving with long context → HBM capacity and bandwidth (H200, MI300X). Cheap high-volume serving → whatever gives the best $/token *after* accounting for ops cost (L40S, Inferentia, Trainium).

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
| 1-bit / 1.58-bit | ~0.13–0.2 | Research (BitNet); surprising but check quality |

**Rule of thumb:** model size in GB ≈ params × bytes-per-param. A 70B model is ~140 GB in bf16, ~70 GB in fp8, ~35 GB in int4.

---

## 4. Memory math (the most useful section in this repo)

### 4.1 Inference memory

```text
weights              = params × bytes_per_param
KV cache             = 2 × n_layers × n_kv_heads × head_dim × seq_len × batch × kv_bytes
activations          = small relative to KV for decode; can be large for prefill of long sequences
overhead             = CUDA context, workspace, fragmentation  → budget 1–3 GB
----------------------------------------------------------------------------
total ≈ weights + KV cache + overhead + (prefill activations)
```

**Worked example — Llama-3-70B (80 layers, 8 KV heads, head_dim 128), bf16:**

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
  gradients             2–4 bytes
  Adam m + v            8 bytes
  -------------------------------------
  ≈ 16–20 bytes/parameter   →  a 7B model needs ~112–140 GB before activations

Plus activations, which scale with batch × sequence × layers × hidden × precision
(this is what activation checkpointing trades compute for).
```

**That's why you need sharding.** ZeRO-1/2/3 or FSDP partitions optimizer states, gradients, and parameters respectively across ranks. A 7B model trains on 8×80 GB because ~140 GB of state is split ~17 GB/GPU.

### 4.3 ZeRO stage cheat

| Stage | Shards | Memory saved | Communication |
|---|---|---|---|
| DDP | nothing | 0 | all-reduce gradients |
| ZeRO-1 | optimizer states | ~8 bytes/param | all-gather + reduce-scatter |
| ZeRO-2 | + gradients | +2–4 bytes/param | + reduce-scatter gradients |
| ZeRO-3 / FSDP | + parameters | up to full model | + all-gather params per layer |

---

## 5. Compute math

### 5.1 Training FLOPs

```text
C_train ≈ 6 × N × D
  N = parameters (non-embedding), D = training tokens

Why 6: 2 (forward) + 4 (backward) multiply-accumulates per param per token.
```

**Cost example — train an 8B model on 1T tokens:**

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
Decode  (memory-bound):   ≈ 2 × N per token     ← but you're waiting on HBM, not FLOPs
```

### 5.3 The decode bandwidth limit (the single best back-of-envelope in serving)

```text
tokens/sec (batch 1) ≤ HBM_bandwidth / weights_bytes

H100: 3.35 TB/s ÷ 16 GB (8B model, bf16) ≈ 209 tok/s theoretical ceiling
                                  ↘ realistic single-stream: 60–120 tok/s
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
| Decode, batch 1 | ~1–2 FLOP/byte | HBM bandwidth |
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
- **Tensor parallel** requires an all-reduce *per layer* — only viable at NVLink speeds, i.e. inside a node.
- **Pipeline parallel** trades idle "bubbles" for less communication; needs enough micro-batches to fill the pipeline.
- **Expert parallel (MoE)** needs all-to-all — the most demanding pattern; it changes your fabric requirements.

**Amdahl's law is the reason for "minimum parallelism that fits":** every added axis adds communication overhead somewhere, so scaling efficiency decays. Measure it (`nccl-tests`, MFU at 1 vs 8 vs 64 GPUs) rather than assuming it.

---

## 8. Cluster design

### Typical node (2026, 8-GPU)

| Component | Typical |
|---|---|
| GPUs | 8× H100/H200/B200 with NVSwitch (900 GB/s–1.8 TB/s each) |
| CPUs | 2× 64-core (Grace/EPYC/Xeon) |
| RAM | 1–2 TB |
| NICs | 8× 400G IB (one per GPU, GPUDirect RDMA) |
| Local NVMe | 4–30 TB |
| Power | ~5.6 kW GPUs + 1.5 kW rest ≈ 7–10 kW/node |

### Sizing rules

| Resource | Rule of thumb |
|---|---|
| Fabric | Non-blocking or rail-optimized for training; some oversubscription is OK for inference |
| Checkpoint storage | Must write your full checkpoint in < 2 minutes; otherwise recovery dominates |
| Dataset throughput | ≥ 1 GB/s per GPU sustained, or the GPUs starve |
| Power | 8×H100 nodes → liquid cooling at rack density; plan for 40–80 kW/rack |
| Failure domains | Assume a GPU failure every few hours at 1,000+ GPUs — design for restart |

### Failure arithmetic

```text
If a GPU has ~20,000 h MTBF, then:
  1,000 GPUs → a failure every ~20 h
 10,000 GPUs → a failure every ~2 h
```

That number is why elastic training, frequent checkpoints, and node quarantine are not optional at scale.

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
| Utilization | sporadic, < 40% | sustained, > 60–70% |
| Ops capacity | small team | dedicated platform team |
| Hardware generation risk | high (churn) | low (stable workload) |
| Data/compliance | cloud-friendly | must stay on-prem |
| Burst capacity | needed | predictable |

Break-even is usually **2–4x** in favor of ownership at high utilization — but only if you can staff the platform. Count the humans.

---

## 10. Numbers worth memorizing

| Quantity | Value |
|---|---|
| Bytes per param: fp32/bf16/fp8/int4 | 4 / 2 / 1 / 0.5 |
| Training FLOPs per token per param | 6 |
| Inference FLOPs per token per param | 2 |
| H100 bf16 dense | ~990 TFLOPS |
| H100 HBM bandwidth | ~3.35 TB/s |
| NVLink per-GPU bandwidth | ~0.9 TB/s (H100) |
| InfiniBand NDR per port | ~50 GB/s |
| H100 FP16/BF16 roofline crossover | ~295 FLOP/byte |
| Chinchilla compute-optimal tokens | ~20 × params |
| Practical tokens/param today | 100–1,000+ (way past Chinchilla) |
| Good MFU for large training | 40–55% |
| Good GPU utilization for serving | 30–70% (idle headroom is a feature) |

---

## 11. Procurement / capacity checklist

Before you buy or reserve anything, write down:

- [ ] **Workload mix** — training vs fine-tuning vs serving, and the ratio
- [ ] **Memory requirement** — largest model, context length, and concurrency
- [ ] **Interconnect requirement** — parallelism plan implies a fabric
- [ ] **Storage requirement** — dataset size, checkpoint size × frequency, bandwidth
- [ ] **Power and cooling** — kW/rack, and whether the building can take it
- [ ] **Utilization forecast** — honest, per quarter, with the idle cost priced in
- [ ] **Ops capacity** — who runs it at 3 a.m.?
- [ ] **Exit plan** — portability (vLLM/K8s/ONNX) so you're never locked in

**Next:** [cheatsheets/inference-math.md](cheatsheets/inference-math.md) for the serving formulas, [cheatsheets/training-parallelism.md](cheatsheets/training-parallelism.md) for the parallelism decision table.
