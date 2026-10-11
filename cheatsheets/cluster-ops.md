# GPU Cluster Ops Cheatsheet

Bring-up, health, scheduling, and incident triage for a GPU fleet. Written for the person on call.

---

## 1. New node bring-up checklist

- [ ] **Drivers**: version match across the fleet; `nvidia-smi` shows all GPUs at expected power/clocks.
- [ ] **Fabric**: IB link state `Active` (`ibstat`), correct rate, no symbol errors.
- [ ] **Topology**: `nvidia-smi topo -m` matches the intended NVLink/PCIe layout.
- [ ] **NCCL**: `nccl-tests` intra-node passes at expected bus bandwidth.
- [ ] **Storage mounts**: dataset and checkpoint paths mounted and writable at expected throughput (`fio` or `dd`).
- [ ] **Container runtime**: `--gpus all` works; the device plugin reports GPUs.
- [ ] **Telemetry**: DCGM exporter → Prometheus; node appears in Grafana.
- [ ] **Burn-in**: 30-60 min of sustained load; watch power, temperature, XID, ECC.
- [ ] **Scheduler**: node joins with correct labels/taints (`gpu=H100`, MIG, topology).
- [ ] **Docs**: node added to inventory with serial numbers and rack position.

> **Rule:** a node is not "ready" because it booted. It's ready when it passed burn-in and appears in telemetry.

---

## 2. Daily health commands

```bash
# GPU inventory + health
nvidia-smi                       # quick view
nvidia-smi -q                    # full detail (ECC, XID, clocks, power)
nvidia-smi dmon -s pucvmet       # live power/util/mem/temp/ECC
nvidia-smi topo -m               # topology (NVLink/PCIe layout)

# Deep health (DCGM)
dcgmi diag -r 2                  # short diagnostic
dcgmi diag -r 3                  # full (takes ~15 min, needs idle GPUs)
dcgmi dmon -e 1001,1002,1003,1004,1005   # GR engine, SM, memory, PCIe, clocks

# InfiniBand
ibstat                           # link state + rate
iblinkinfo                       # topology
perfquery                        # error counters (watch for growth)

# Fabric performance
./all_reduce_perf -b 8 -e 8G -f 2 -g 8
```

---

## 3. The metrics that matter

| Signal | Why | Alarm if |
|---|---|---|
| SM utilization | Are we using the compute we paid for? | Sustained < 20% on a busy node |
| HBM utilization | Memory-bandwidth bound? | Pinned at 100% with low SM |
| GPU power draw | Proxy for real work | At idle levels during a job |
| GPU memory used | Capacity + leak detection | Creeping upward across steps |
| Temperature | Thermal throttling | > 85 °C sustained |
| Clock throttle reasons | Silently slow hardware | `sw_power_cap` / `hw_slowdown` |
| XID errors | Driver/hardware faults | Any new XID |
| ECC (corrected/uncorrected) | Memory health | Any uncorrected; rising corrected |
| NVLink error counters | Interconnect health | Any growth |
| IB symbol/port errors | Fabric health | Any growth |
| Per-rank step time vs median | **Fail-slow / straggler detection** | Any rank > 1.2x the median, sustained |
| Job queue depth | Capacity pressure | Growing wait times |

---

## 4. Scheduling quick reference

### Slurm

```bash
squeue -u $USER                       # my jobs
squeue -p gpu -t R                    # running jobs on a partition
sacct -j <jobid> --format=JobID,Elapsed,AllocCPUS,ReqGRES,MaxRSS,State
sinfo -o "%20N %10c %10m %25f %10G"   # partition/GRES view
scancel <jobid>
scontrol show node <node>             # node state, GRES, features
```

**Best practices:** always request `--gres=gpu:N`; set `--time` realistically (it affects backfill); use `--constraint` for topology/site; never run heavy work on the login node.

### Kubernetes

```bash
kubectl get nodes -L nvidia.com/gpu.product
kubectl describe node <node> | grep -A10 "Allocated resources"
kubectl get pods -A -o wide --field-selector spec.nodeName=<node>
kubectl top pods -A
kubectl get jobs,workloads -A
kubectl logs <pod> -c <container> --tail=200
kubectl get events -A --sort-by=.lastTimestamp | tail -50
```

**Common failure modes**

| Symptom | Cause | Fix |
|---|---|---|
| Pod stuck `Pending` | No free GPU / quota exceeded / unsatisfiable selector | Check Kueue quota and node labels |
| Pod `Running` but no GPU | Missing device plugin or DRA claim | Check `nvidia.com/gpu` allocation |
| Job evicted | Preemption or node pressure | Check priority class and quota |
| GPU not visible in container | Driver/toolkit mismatch | Reinstall GPU Operator components |

---

## 5. Storage & checkpoint hygiene

- **Checkpoint write bandwidth** must let you save the full checkpoint in < 2 minutes. Measure it, don't assume it.
- **Never write checkpoints directly to the object store** if latency matters; stage on parallel FS or local NVMe, then copy.
- **Version your datasets**; a silent dataset change invalidates every result.
- **Watch the many-small-files problem**: shard datasets into large files before training.
- **Alert on storage saturation**, not on storage failure. Full or slow storage kills runs faster than dead disks.

---

## 6. Incident triage runbook

### "Training job is stuck / hung"

1. `nvidia-smi` everywhere. Any GPU at 0% while others are busy is your suspect.
2. Check the rendezvous/master address reachability and firewall.
3. `NCCL_DEBUG=INFO` on a short reproduction: is IB being used?
4. Enable the flight recorder; dump the trace buffer on timeout.
5. Check for a slow rank (per-rank step times), network error counters, and thermal throttling.
6. If a single GPU is bad: drain the node, quarantine it, restart the job.

### "Loss is NaN"

1. Check LR, warmup, and grad clipping.
2. Check dtype: bf16 is more forgiving than fp16 without loss scaling.
3. Check the data: any malformed or empty samples?
4. Reproduce at small scale; bisect the config.

### "Throughput dropped 30% overnight"

1. Thermal throttling? Check clocks and temps.
2. Noisy neighbor on the same node/NIC/switch?
3. Storage contention during peak hours?
4. A code/config change? Use git + config hashes.
5. A degraded GPU whose clocks silently dropped?

### "Job is running but much slower than it should be" (fail-slow)

This is the most expensive and least visible failure mode on a GPU fleet, and it emits **no error at all**. A straggler GPU passes every health check while dragging the whole job. Studies of production clusters find most large jobs hit at least one straggler, producing tens of percent completion delay.

Detection is **relative, never absolute**:

1. Log per-rank step time; alert on **outliers vs the fleet median** (z-score), not on absolute thresholds.
2. Per-GPU achieved TFLOPS or tokens/s during the run — a rank at 60% of median is your problem.
3. Re-run `nccl-tests` on the suspect node and compare bus bandwidth against a known-good node.
4. Check NVLink error counters and IB symbol errors on that specific node, not just cluster-wide totals.
5. Check for clock throttling (`nvidia-smi -q -d PERFORMANCE`) that never tripped a thermal alarm.
6. **Then** drain and re-test with `dcgmi diag -r 3`.

> **Why this matters:** a clean fail-stop is cheap — the job dies, you restart, you lose minutes. A fail-slow job burns the *full* allocation while delivering half the work, and it typically only surfaces as "this run took 3 days instead of 2." Instrument for it explicitly, or you will pay for it without noticing.

### "A GPU failed mid-run"

1. Identify via XID / ECC / DCGM.
2. Confirm with `dcgmi diag -r 3` on an idle GPU.
3. If bad: cordon the node, record the serial, open an RMA, document in the fleet log.
4. Add the failure mode to the runbook. Repeat failures on the same model/rack are a pattern, not a coincidence.

---

## 7. The cost view you should always have open

| Dashboard | Metric |
|---|---|
| Fleet utilization | Allocated vs actually used GPU-hours |
| Idle waste | Idle GPU-hours × $/hour, per team |
| Job efficiency | MFU for training, goodput for serving |
| Queue wait | Time-to-start by priority class |
| Cost per output | $/1M tokens per endpoint |
| Cost per run | $ per training run, trended over time |

Dashboards that don't change a decision are decoration. Each of these should map to an action someone owns.

---

## 8. Weekly hygiene

- [ ] Review new XID/ECC events; retire anything trending bad.
- [ ] Check IB error counters per switch port.
- [ ] Verify checkpoints are restorable (restore one on purpose).
- [ ] Reconcile utilization against the bill.
- [ ] Retire or fix nodes below a utilization threshold.
- [ ] Re-run `dcgmi diag` on the least-used nodes (rolling, on idle hardware).

---

**Back to:** [README](../README.md) · [ROADMAP](../ROADMAP.md) · [HARDWARE](../HARDWARE.md) · [training parallelism](training-parallelism.md)
