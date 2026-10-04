# 🧮 Training & Parallelism Cheatsheet

How to fit a model, how to shard it, and how to debug it when it hangs.

---

## 1. Will it fit?

```text
Mixed-precision Adam, per parameter:
  bf16 weights        2 B
  fp32 master         4 B
  gradients           2–4 B
  Adam m, v           8 B
  ----------------------------------
  ≈ 16–20 B/param   (before activations)

Full 7B  → ~112–140 GB     Full 70B → ~1.1–1.4 TB
```

| Strategy | 7B on 8×80 GB | Notes |
|---|---|---|
| DDP (replicated) | ✗ | 140 GB of state per GPU |
| ZeRO-1 | ✓ | Shards optimizer states |
| ZeRO-2 | ✓✓ | + gradients |
| ZeRO-3 / FSDP | ✓✓✓ | + parameters; slowest comms |

**Also add activations** (scales with batch × seq × layers × hidden). Activation checkpointing trades ~30% compute for large memory savings.

---

## 2. Parallelism decision table

| Axis | What it splits | Communication | Use when |
|---|---|---|---|
| **DP** | Batch | All-reduce gradients (per step) | Model fits; scale throughput |
| **ZeRO/FSDP** | Optimizer/grad/params | All-gather + reduce-scatter | Model doesn't fit replicated |
| **TP** | Layers/matrices | All-reduce **per layer** | Model doesn't fit on one GPU; **intra-node only** |
| **PP** | Layer stages | Activations between stages | Very large models; needs micro-batches |
| **SP** | Activations (sequence) | Small, paired with TP | Reduce activation memory |
| **CP** | Long sequences | Ring attention comms | Context too long for one GPU |
| **EP** | MoE experts | **All-to-all** | MoE models |

### Choosing (rule of thumb)

```text
1. Fit the model with the FEWEST axes possible.
2. TP ≤ GPUs per node (NVLink). Never TP across PCIe.
3. Add PP (across nodes) when the model exceeds one node even at full TP.
4. Use DP/FSDP for the remaining scale-out.
5. EP only for MoE; budget the network for all-to-all.
```

**Prefer FSDP over hand-rolled ZeRO-3** in pure PyTorch. **Prefer Megatron/torchtitan** when you need peak MFU and are willing to invest.

---

## 3. Communication cost

```text
All-reduce bytes per rank ≈ 2 × (N−1)/N × S      (S = payload)

Consequences:
  • Larger messages → bandwidth-bound (good, efficient)
  • Small messages  → latency-bound (bad, overhead dominates)
  • TP does an all-reduce per layer → wants NVLink
  • MoE all-to-all is the most demanding pattern
```

**Scaling efficiency** = (tokens/s at N GPUs) ÷ (N × tokens/s at 1 GPU). Report it; don't assume it.

---

## 4. Hyperparameters that scale with the plan

| Knob | Rule |
|---|---|
| Global batch size | micro-batch × grad-accum × DP size |
| LR | Scale sub-linearly with global batch; add warmup |
| Warmup | ~0.1–1% of total steps for large runs |
| Grad accumulation | Memory lever; keep the *global* batch constant when you change it |
| Sequence length | Quadratic attention cost; budget it |
| Weight decay | ~0.1 typical; skip on norms/biases |
| Checkpoint interval | Must satisfy your RTO; measure write time, don't guess |

> **Reproducibility:** fix seeds, pin library versions, log the full config, and record the exact global batch size. "It trained differently" is usually a config drift.

---

## 5. Config templates

### Single node, 8 GPUs, model fits (DDP)

```bash
torchrun --nproc_per_node=8 train.py \
  --bf16 --batch-size 8 --grad-accum 4 \
  --strategy ddp
```

### Single node, 8 GPUs, model doesn't fit (FSDP)

```bash
accelerate launch --num_processes 8 train.py \
  --mixed_precision bf16 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_config fsdp_config.json
```

### Two nodes × 8 GPUs (FSDP + HSDP)

```bash
# node 0
torchrun --nnodes=2 --node_rank=0 --master_addr=$HEAD \
         --nproc_per_node=8 train.py --strategy fsdp

# node 1
torchrun --nnodes=2 --node_rank=1 --master_addr=$HEAD \
         --nproc_per_node=8 train.py --strategy fsdp
```

Set `NCCL_DEBUG=INFO`, `NCCL_IB_HCA`, and `NCCL_SOCKET_IFNAME` correctly *before* you debug anything else.

### Slurm

```bash
#!/bin/bash
#SBATCH --job-name=llm-train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out

srun --kill-on-bad-exit=1 torchrun \
  --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID \
  --nproc_per_node=8 --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py --strategy fsdp
```

---

## 6. Debugging ladder (in order)

| Step | Check |
|---|---|
| 1 | `nvidia-smi` on every node — are all GPUs visible and healthy? |
| 2 | `NCCL_DEBUG=INFO` — is it picking IB or falling back to sockets? |
| 3 | `nccl-tests all_reduce_perf` — does the fabric perform as expected? |
| 4 | Fix the seed and re-run with 1 GPU — does it work at all? |
| 5 | Scale 1 → 2 → 8 GPUs, watching MFU at each step |
| 6 | Enable the NCCL flight recorder / `TORCH_NCCL_TRACE_BUFFER_SIZE` to catch hangs |
| 7 | Check for a straggler: per-rank step times, not just the mean |

**Common causes of hangs:** mismatched `MASTER_ADDR/PORT`, firewall blocking the rendezvous, a rank with a different batch size, an imbalanced dataset shard, or `NCCL_SOCKET_IFNAME` pointing at the wrong NIC.

---

## 7. Efficiency checklist

- [ ] Data loading isn't the bottleneck (check GPU idle time during training).
- [ ] Compute/comm overlap enabled (FSDP `backward_prefetch`, TP overlap).
- [ ] Activation checkpointing applied only where needed.
- [ ] FlashAttention enabled (or an equivalent fused attention).
- [ ] `torch.compile` tried for the hot path (and measured).
- [ ] Batch size large enough to be compute-bound.
- [ ] MFU recorded and compared across configs.

**MFU targets:** > 40% is solid for large multi-node training; > 50% is excellent. Below 30% means you're paying for GPUs you aren't using.

---

**Back to:** [README](../README.md) · [HARDWARE](../HARDWARE.md) · [inference math](inference-math.md) · [cluster ops](cluster-ops.md)
