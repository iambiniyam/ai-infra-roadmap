# ⚡ Inference Math Cheatsheet

Everything you need to size, tune, and price an LLM deployment. Print this.

---

## 1. Memory

```text
Total VRAM ≈ weights + KV cache + activations + overhead(~1–3 GB)

weights       = params × bytes_per_param
KV cache      = 2 × n_layers × n_kv_heads × head_dim × seq_len × batch × kv_bytes
prefill acts  ≈ batch × seq_len × hidden × precision_bytes × (small factor)
```

| Precision | bytes/param | 7B | 70B | 400B |
|---|---:|---:|---:|---:|
| FP32 | 4 | 28 GB | 280 GB | 1.6 TB |
| BF16/FP16 | 2 | 14 GB | 140 GB | 800 GB |
| FP8/INT8 | 1 | 7 GB | 70 GB | 400 GB |
| INT4 | 0.5 | 3.5 GB | 35 GB | 200 GB |

**KV cache per token** for common configs (bf16, 1 sequence):

| Model | layers | kv_heads | head_dim | bytes/token | @8k | @32k | @128k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3-8B | 32 | 8 | 128 | 0.13 MB | 1.1 GB | 4.3 GB | 17 GB |
| Llama-3-70B | 80 | 8 | 128 | 0.33 MB | 2.6 GB | 10.5 GB | 42 GB |
| Mistral-7B (MHA) | 32 | 32 | 128 | 0.52 MB | 4.3 GB | 17 GB | 68 GB |

> Multiply by batch size. **Long context × high concurrency is what forces FP8 KV, prefix caching, or disaggregation.**

---

## 2. Speed

```text
Decode ceiling (batch 1)  ≈ HBM_bandwidth ÷ weights_bytes
                        ≈ 3.35 TB/s ÷ 16 GB ≈ 209 tok/s   (8B bf16 on H100)

Realistic single-stream: 50–70% of that → 100–150 tok/s
With batching, throughput ≈ ceiling × batch (until compute-bound)
```

**The three regimes:**

| Regime | Bottleneck | Fix |
|---|---|---|
| Decode, small batch | HBM bandwidth | Quantize, bigger batch, speculative decode |
| Prefill, long prompt | Compute | Chunked prefill, better kernels, FP8 |
| Decode, large batch | Compute + KV bandwidth | Quantized KV, paged attention, more replicas |

---

## 3. The metrics you report

| Metric | Definition | Target intuition |
|---|---|---|
| **TTFT** | Time to first token (≈ queue + prefill) | < 300 ms feels instant; < 1 s tolerable |
| **TPOT / ITL** | Time per output token | < 30 ms ≈ faster than reading speed |
| **Throughput** | Total tokens/s across requests | The GPU-efficiency number |
| **Goodput** | Tokens/s *meeting the SLO* | The number to optimize |
| **p95 / p99** | Tail latencies | What users actually complain about |
| **$/1M tokens** | Cost per million tokens in/out | The business number |
| **GPU util** | SM occupancy / HBM utilization | 30–70% for serving is healthy |

---

## 4. Cost

```text
$/1M tokens = (GPU_hourly_cost × 1e6) ÷ (tokens_per_second × 3600)

Example: 2×H100 node @ $5.00/hr, 1,500 output tok/s
       = (5.00 × 1e6) ÷ (1500 × 3600) = $0.93 per 1M output tokens
```

**Lever order (biggest first):**

1. **Right-size the model** — a smaller model that passes evals beats a big one that doesn't pay for itself.
2. **Quantize** — FP8/AWQ/INT4 reduces bytes ⇒ faster decode and more batch headroom.
3. **Batch** — biggest throughput lever; costs latency.
4. **Prefix caching** — eliminates redundant prefill on shared system prompts / few-shot.
5. **Quantized KV cache** — more concurrent long contexts per GPU.
6. **Speculative decoding** — lower latency per token at slightly higher compute.
7. **Disaggregation** — better TTFT under long-prompt traffic; not always a win.
8. **Hardware mix** — cheap cards for cheap models; don't brute-force with H100s.

---

## 5. Tuning order (do it in this sequence)

1. **Model + precision** — pick the smallest model that passes evals; start at bf16, then quantize.
2. **`max_model_len`** — set to what you actually need. Context is memory; memory is cost.
3. **`max_num_seqs` / batch** — raise until the latency SLO is at risk, then back off ~20%.
4. **KV cache dtype** — FP8 if quality tolerates it.
5. **Prefix caching** — enable if prompts share prefixes (they usually do).
6. **Chunked prefill** — enable for long prompts to protect ITL.
7. **Speculative decoding** — only after the above.
8. **Replicas + routing** — horizontal scaling, cache-aware load balancing.

At every step: **measure one change at a time, and keep the config pinned in git.**

---

## 6. Choosing a parallelism plan for inference

```text
Does the model fit on one GPU with room for KV?
├─ yes → single GPU (simplest, best latency)
└─ no  → split the model. TP within a node (NVLink), never across PCIe.
         ├─ does it fit in one node after TP? → TP=N, single node
         └─ no → TP within node + PP across nodes (or disaggregated)
```

| Deployment | Best for |
|---|---|
| Single GPU | Small models, low latency, simple ops |
| TP within a node | Large models, latency-sensitive |
| TP + PP across nodes | Very large models, throughput-oriented |
| Data-parallel replicas + router | High QPS with a small model |
| Disaggregated prefill/decode | Mixed traffic with long prompts |
| Expert parallel (MoE) | MoE models; requires fast all-to-all |

---

## 7. Quick benchmark hygiene

- [ ] Warm up (≥ 20 requests) before measuring.
- [ ] Fix the prompt distribution (input/output length percentiles).
- [ ] Pin versions: model, engine, driver, CUDA, GPU.
- [ ] Report p50 **and** p95/p99, never just the mean.
- [ ] Sweep concurrency: 1, 4, 16, 64, 128, 256, 512.
- [ ] Record GPU power and clock state (throttling ruins comparisons).
- [ ] Save raw results to a file; plots are for humans, files are for truth.

---

## 8. Failure signatures

| Symptom | Likely cause |
|---|---|
| TTFT spikes with long prompts | Prefill compute/queueing; enable chunked prefill |
| ITL degrades as concurrency rises | Batch too large or KV bandwidth-bound |
| Throughput plateaus early | Compute-bound; check dtype, kernel selection |
| OOM only at high concurrency | KV cache growth; cap `max_num_seqs` or quantize KV |
| First request slow, later fast | Cold start / no warm pool |
| Throughput drops over hours | Thermal throttling, memory fragmentation, or a leak |
| Big gap between 1 GPU and TP=2 | PCIe TP or bad collective overlap |

---

**Back to:** [README](../README.md) · [ROADMAP](../ROADMAP.md) · [HARDWARE](../HARDWARE.md) · [training parallelism](training-parallelism.md)
