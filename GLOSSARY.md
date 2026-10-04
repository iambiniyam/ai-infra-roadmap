# Glossary

Over 200 terms, one line each, grouped by layer. Alphabetical within groups.

---

## Hardware & systems

- **Accelerator**: Any chip that runs model math fast: GPU, TPU, NPU, FPGA, ASIC.
- **HBM**: High Bandwidth Memory, the GPU's main memory. Bandwidth, not capacity, usually sets inference speed.
- **SM**: Streaming Multiprocessor. The GPU's compute unit; "SM utilization" is the headline GPU metric.
- **SRAM**: On-chip scratchpad memory (very fast, tiny). The resource FlashAttention is designed around.
- **L2 cache**: Chip-wide cache between SRAM and HBM; ~40-60 MB on modern GPUs.
- **Roofline model**: A plot relating arithmetic intensity to achievable performance; tells you if a kernel is compute- or memory-bound.
- **Arithmetic intensity**: FLOPs performed per byte moved. Low intensity means memory-bound.
- **Memory-bound**: Performance limited by data movement, not math. Most LLM decode is this.
- **Compute-bound**: Performance limited by math throughput. Big GEMMs and prefill are this.
- **TFLOPS**: Tera floating-point operations per second. Always ask *which precision* and *dense or sparse*.
- **MFU**: Model FLOPs Utilization: achieved model FLOPs ÷ peak hardware FLOPs. The honest efficiency metric for training.
- **FLOPS vs FLOPs**: FLOPS = rate (per second); FLOPs = a count (total work). Be precise; people are sloppy here.
- **PCIe**: The host↔device bus. Fast enough for data loading, too slow for tensor parallelism.
- **NVLink**: NVIDIA's high-speed GPU↔GPU interconnect (~900 GB/s on H100). Where TP lives.
- **NVSwitch**: The intra-node switch giving every GPU non-blocking access to every other GPU.
- **InfiniBand (IB)**: The dominant inter-node fabric; NDR ≈ 50 GB/s per port.
- **RoCE**: RDMA over Converged Ethernet. Cheaper fabric; needs careful tuning.
- **RDMA**: Remote Direct Memory Access. Zero-copy, kernel-bypass networking.
- **GPUDirect RDMA**: Lets a NIC write directly into GPU memory, skipping the CPU.
- **Fat-tree**: A common non-blocking network topology for clusters.
- **Rail-optimized**: Wiring where each GPU talks to a matching NIC, minimizing hops.
- **Oversubscription**: Network links share bandwidth (e.g. 3:1). Raises cost efficiency, lowers worst-case bandwidth.
- **MIG**: Multi-Instance GPU: partition one physical GPU into isolated slices.
- **Time-slicing**: Sharing a GPU by context switching. Cheap, but workloads interfere.
- **NUMA**: Non-uniform memory access; CPU/memory locality that matters for data loading.
- **TDP / power cap**: Thermal design power; capping power trades throughput for watts.
- **PUE**: Power Usage Effectiveness of a datacenter (~1.1 typical, lower with liquid cooling).

---

## Precision & numerics

- **FP32**: 32-bit float. Legacy training default; 4 bytes.
- **TF32**: 19-bit-mantissa format used implicitly for matmuls on Ampere+; a free speedup.
- **BF16**: 16-bit brain float. Wide dynamic range; the modern training/serving default.
- **FP16**: 16-bit float. More mantissa, less range; needs loss scaling.
- **FP8**: 8-bit float (E4M3/E5M2). Big bandwidth win on Hopper/Blackwell; watch outliers.
- **FP4 / NVFP4**: 4-bit float with a shared scale; Blackwell-era inference.
- **INT8 / INT4**: Integer quantization; used in GPTQ/AWQ/GGUF.
- **Quantization**: Reducing numeric precision to save memory/bandwidth. Quality is the trade.
- **Calibration**: Choosing quantization scales from representative data.
- **PTQ vs QAT**: Post-training quantization (no retraining) vs quantization-aware training (better quality, more work).
- **Loss scaling**: Multiplying loss to keep FP16 gradients in range.
- **Numerical stability**: Whether a run diverges from rounding error. Usually about softmax, layernorm, or attention logits.
- **Outlier**: A few extreme activations that wreck naive quantization.
- **Perplexity**: Exponentiated average negative log-likelihood; a cheap language-model quality proxy.

---

## Model architecture

- **Transformer**: The attention-based architecture underlying essentially all modern LLMs.
- **Attention**: The mechanism that lets tokens weigh other tokens. Quadratic in sequence length.
- **Self-attention**: Attention of a sequence with itself.
- **MHA / MQA / GQA**: Multi-head, multi-query, and grouped-query attention. Fewer KV heads means a smaller KV cache.
- **KV cache**: Cached keys/values from prior tokens so decode doesn't recompute the past. Dominates serving memory.
- **FlashAttention**: IO-aware exact attention that tiles through SRAM instead of materializing the score matrix.
- **MoE**: Mixture of Experts: route tokens to a subset of experts. Big capacity, all-to-all communication.
- **Expert parallelism (EP)**: Sharding MoE experts across devices.
- **Router / gating**: The network deciding which experts a token visits.
- **MLA**: Multi-head Latent Attention (DeepSeek); compresses the KV cache.
- **Speculative decoding**: Draft tokens cheaply, verify in parallel. Faster decode at the same output distribution.
- **Medusa / EAGLE**: Popular speculative-decoding head designs.
- **Context window**: Maximum tokens a model can attend to at once.
- **RoPE / ALiBi**: Positional encoding schemes.
- **Token**: A subword unit; the unit of billing, throughput, and context length.
- **BPE**: Byte-Pair Encoding; the dominant tokenizer algorithm.
- **Chat template**: The exact role/message formatting a model was trained on.

---

## Training & parallelism

- **Pretraining**: Training from random weights on a large corpus.
- **Fine-tuning**: Adapting a pretrained model on task data.
- **SFT**: Supervised fine-tuning on demonstrations.
- **PEFT**: Parameter-efficient fine-tuning (LoRA et al.).
- **LoRA**: Trains low-rank adapter matrices; freezes base weights. Cheap and composable.
- **QLoRA**: LoRA on a 4-bit-quantized base model. Fits big models on small GPUs.
- **DoRA**: A LoRA variant with magnitude/direction decomposition.
- **RLHF**: Reinforcement learning from human feedback (reward model + PPO).
- **DPO**: Direct Preference Optimization; preference tuning without an RL loop.
- **GRPO**: Group Relative Policy Optimization; the workhorse of open reasoning-model training.
- **RLVR**: Reinforcement learning with verifiable rewards (math, code).
- **Reward model**: A model scoring outputs, used as the RL objective.
- **Distillation**: Training a small model to imitate a large one (logits or data).
- **Data parallel (DP)**: Replicate the model; split the batch; all-reduce gradients.
- **DDP**: PyTorch's data-parallel wrapper.
- **Tensor parallel (TP)**: Split individual layers/matrices across devices. Needs fast interconnect.
- **Pipeline parallel (PP)**: Split layers into stages across devices; introduces bubbles.
- **Sequence parallel (SP)**: Shard the sequence dimension of activations.
- **Context parallel (CP)**: Shard long sequences across devices (ring attention).
- **ZeRO-1/2/3**: Shard optimizer states / gradients / parameters (progressively).
- **FSDP**: Fully Sharded Data Parallel; PyTorch's ZeRO-3 equivalent.
- **HSDP**: Hybrid sharded data-parallel: shard within a node, replicate across.
- **Gradient accumulation**: Sum gradients over micro-batches to simulate a larger batch.
- **Micro-batch**: One forward/backward pass; the unit you actually fit in memory.
- **Global batch size**: Micro-batch × accumulation × data-parallel size.
- **Activation checkpointing**: Recompute activations in backward to save memory.
- **Optimizer state**: Adam's moment estimates; usually the largest training memory consumer.
- **Warmup / cosine schedule**: LR schedules that keep training stable.
- **Gradient clipping**: Bounds gradient norms to prevent divergence.
- **Divergence**: Loss blows up; usually LR, dtype, or data.
- **Loss spike**: A transient divergence that sometimes self-corrects.
- **Checkpoint**: Serialized model/optimizer state for resumption.
- **Sharded checkpoint**: Checkpoint split across ranks; needs reassembly logic.
- **Elastic training**: Changing the number of workers mid-run.

---

## Inference & serving

- **Prefill**: Processing the input prompt. Compute-bound; sets TTFT.
- **Decode**: Generating tokens one at a time. Memory-bound; sets inter-token latency.
- **PagedAttention**: Storing KV cache in non-contiguous blocks, enabling sharing and less fragmentation.
- **Continuous batching**: Adding/removing requests from a batch every step instead of waiting for a full batch.
- **Static batching**: Waiting for a fixed batch; poor utilization for chat.
- **Chunked prefill**: Splitting long prefills to interleave with decode and bound latency.
- **Prefix caching**: Reusing KV cache for shared prompt prefixes (system prompts, few-shot examples).
- **RadixAttention**: Prefix caching implemented as a radix tree (SGLang).
- **TTFT**: Time To First Token. The perceived "thinking" latency.
- **TPOT / ITL**: Time Per Output Token / Inter-Token Latency. The perceived typing speed.
- **Throughput**: Tokens processed per second across all requests.
- **Goodput**: Throughput that meets the SLO. The metric that matters.
- **p50/p95/p99**: Percentile latencies; tail latency is what users complain about.
- **Batch size**: Requests processed together; higher means more throughput and worse latency.
- **max_num_seqs**: The engine's concurrency cap. The main throughput/latency knob.
- **Disaggregation**: Separating prefill and decode onto different workers.
- **KV transfer**: Moving KV cache between workers (NIXL, etc.) in disaggregated serving.
- **Quantized KV cache**: Storing the cache at FP8/INT8 to fit longer contexts.
- **Sampling**: temperature, top-p, top-k, repetition penalty; changes output diversity, not speed.
- **Structured output / grammar**: Constrained decoding to valid JSON/regex/grammar.
- **Serving framework**: vLLM, SGLang, TensorRT-LLM, TGI, LMDeploy.
- **Model server**: Triton, TorchServe, BentoML, KServe; wraps models as services.
- **Autoscaling**: Adding/removing replicas by load or queue depth.
- **Cold start**: Time from scale-up to ready; dominated by model load.
- **Warm pool**: Pre-loaded replicas to hide cold starts.

---

## Data & storage

- **Object storage**: S3-style key/blob store. Cheap, durable, high-latency.
- **POSIX FS**: Traditional file semantics; needed by most training code.
- **Parallel FS**: Distributed file system built for aggregate bandwidth (Lustre, GPFS, JuiceFS).
- **Lakehouse**: Object storage + a table format giving ACID and schema (Iceberg/Delta/Hudi).
- **Table format**: Iceberg, Delta, Hudi, Lance.
- **Columnar**: Storage layout reading only needed columns (Parquet, Arrow).
- **Parquet / ORC**: Columnar file formats.
- **Arrow**: In-memory columnar format; the interchange standard.
- **Sharding**: Splitting data into many large files. The cure for the many-small-files problem.
- **Many-small-files problem**: Millions of tiny files destroy metadata and throughput.
- **WebDataset / Mosaic Streaming**: Tar-shard-based formats built for training throughput.
- **Feature store**: Consistent features between training and serving (Feast).
- **Lineage**: Tracking where data/models came from and what derived from what.
- **Data versioning**: DVC, lakeFS, or content-addressed storage.
- **ETL / ELT**: Extract-transform-load vs load-then-transform.
- **Batch vs streaming**: Periodic vs continuous processing.
- **CDC**: Change Data Capture; streaming database changes.
- **Chunking**: Splitting documents for retrieval. The most underrated quality lever in RAG.

---

## Orchestration & platform

- **Scheduler**: Assigns jobs to resources (Slurm, K8s, Kueue, Volcano).
- **Gang scheduling**: All-or-nothing placement of a distributed job's pods.
- **Backfill**: Filling idle capacity with later, smaller jobs.
- **Preemption**: Evicting lower-priority work for higher-priority work.
- **Quota**: Hard or soft resource limit per team/namespace.
- **Fair sharing**: Dividing resources by entitlement and usage (DRF-style).
- **K8s**: Kubernetes; the container orchestration standard.
- **Pod / Deployment / Job**: K8s workload primitives.
- **DaemonSet**: Runs one pod per node (e.g. the GPU device plugin).
- **Operator**: Custom controller that manages complex apps (GPU Operator).
- **CRD**: Custom Resource Definition; how K8s is extended.
- **Device plugin**: Exposes vendor resources (GPUs) to the kubelet.
- **DRA**: Dynamic Resource Allocation; the modern K8s way to request accelerators.
- **Kueue**: K8s-native job queueing and quota management.
- **Volcano**: CNCF batch scheduler with gang scheduling.
- **JobSet / LeaderWorkerSet**: K8s APIs for multi-pod distributed jobs.
- **Ray head node**: Ray's coordinator; a single point to watch.
- **SkyPilot**: Portable job placement across clouds, K8s, and Slurm.
- **Slurm**: HPC workload manager; `sbatch`/`srun`, GRES, topology-aware.
- **Topology-aware scheduling**: Placing jobs to minimize network hops.
- **Blue/green & canary**: Deployment strategies for safe model rollout.
- **GitOps**: Declarative, git-driven cluster state (Argo CD).
- **IaC**: Infrastructure as Code (Terraform, Pulumi, Ansible).

---

## Retrieval, context & agents

- **Embedding**: Dense vector representation of content.
- **ANN**: Approximate Nearest Neighbor search (HNSW, IVF, PQ).
- **HNSW**: Graph-based ANN index; the common default.
- **IVF / PQ**: Inverted file / product quantization; memory-efficient ANN.
- **Recall@k**: Fraction of true neighbors found among the top k.
- **Hybrid search**: Combining lexical (BM25) and dense retrieval.
- **BM25**: Classic lexical ranking function; still hard to beat.
- **Reranker**: A cross-encoder that reorders retrieved candidates for quality.
- **RAG**: Retrieval-Augmented Generation.
- **GraphRAG**: Retrieval over a knowledge graph of entities and relations.
- **Context engineering**: Curating what goes in the window; the practical successor to prompt engineering.
- **Prompt injection**: Untrusted content hijacking an agent's instructions.
- **Agent**: An LLM in a loop with tools, state, and a goal.
- **Tool calling**: Model-issued structured calls to functions/APIs.
- **MCP**: Model Context Protocol; standard for exposing tools/context.
- **A2A**: Agent-to-Agent protocol for inter-agent interop.
- **ReAct**: Reason-and-act loop pattern.
- **Sandbox**: Isolated environment to run model-generated code (E2B, Daytona, Firecracker).
- **Durable execution**: Persisted, resumable workflow state (Temporal).
- **Human-in-the-loop**: Approval gates inside automated workflows.
- **Guardrail**: Runtime check that blocks or clips unsafe/invalid output.

---

## Observability, evaluation & operations

- **SLI / SLO / SLA**: Indicator, objective, agreement. What you measure, target, and promise.
- **Error budget**: Allowed unreliability before you must stop shipping.
- **Tracing**: End-to-end record of a request across components.
- **Span**: One unit of work inside a trace.
- **OpenTelemetry**: Vendor-neutral standard for traces/metrics/logs.
- **Eval**: Systematic quality measurement of a model or system.
- **LLM-as-judge**: Using a model to score outputs. Cheap and biased; calibrate it.
- **Golden set**: A curated, versioned eval dataset.
- **Regression test**: An eval that fails when quality drops.
- **Hallucination**: Confidently wrong output.
- **Faithfulness / groundedness**: Whether output is supported by retrieved context.
- **Drift**: Distribution change in inputs, outputs, or data.
- **Red-teaming**: Adversarially probing a system for failures (garak, promptfoo).
- **Jailbreak**: Bypassing a model's safety behavior.
- **PII / PHI**: Personally identifiable / protected health information.
- **MTTR / MTBF**: Mean time to recovery / between failures.
- **RTO / RPO**: Recovery time objective / recovery point objective.
- **Postmortem**: Blameless incident review.
- **Burn-in**: Stress-testing new hardware before trusting it.
- **XID error**: NVIDIA GPU/driver error code; the fleet-health signal.
- **ECC error**: Memory error rate on a GPU; rising counts mean retire the card.
- **FinOps**: Managing cloud/GPU cost as an engineering discipline.
- **Showback / chargeback**: Reporting cost per team / actually billing it.

---

**Back to:** [README](README.md) · [ROADMAP](ROADMAP.md) · [STACK](STACK.md) · [cheatsheets/](cheatsheets/)
