# The Stack

Over 200 open-source projects, organized by the [9 layers](README.md#the-map). Star counts and licenses were pulled from the GitHub API on **2026-10-11** and are refreshed weekly by [`scripts/refresh_stars.py`](scripts/refresh_stars.py).

**How to read this:** each layer starts with a short "decision notes" block, the mental model for choosing, followed by the table. `Stars` column is a rough popularity signal, **not** a quality score. A 2k-star project that solves your exact problem beats a 90k-star project that doesn't.

> **License note:** anything marked `NOASSERTION` or `AGPL-3.0`/`GPL-3.0` needs a legal look before you embed it in a product. `SSPL`/`BSL`/dual-license projects (MinIO, Redis, Terraform, and some others) are *source-available*, not OSI-open-source. They are fine to use, but know the difference. This is not legal advice.

---

## L0 · Hardware & Facility

**Decision notes.** You rarely choose hardware in code, but it dictates everything above it. Two questions decide most designs: *how fast can I move bytes into the chip* (HBM bandwidth) and *how fast can chips talk to each other* (NVLink intra-node, InfiniBand inter-node). Everything else is second-order. See [HARDWARE.md](HARDWARE.md) for the math.

| Project | Stars | License | Notes |
|---|---:|---|---|
| [NVIDIA/gpu-operator](https://github.com/NVIDIA/gpu-operator) | 2,924 | Apache-2.0 | Turns a K8s node into a GPU node: drivers, container toolkit, DCGM, device plugin, MIG. Start here on K8s. |
| [NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) | 3,894 | Apache-2.0 | Exposes GPUs to the kubelet. Understand it when pods can't see GPUs. |
| [NVIDIA/DCGM](https://github.com/NVIDIA/DCGM) | 799 | Apache-2.0 | GPU telemetry and health: utilization, ECC, XID, clocks, power. The source of truth for GPU fleets. |
| [ROCm/ROCm](https://github.com/ROCm/ROCm) | 6,819 | MIT | AMD's CUDA alternative. Real option for inference; check model coverage before committing. |
| [aws-neuron/aws-neuron-sdk](https://github.com/aws-neuron/aws-neuron-sdk) | 640 | NOASSERTION | Trainium/Inferentia. Best $/token for some workloads; narrow op coverage. |
| [NVIDIA/cccl](https://github.com/NVIDIA/cccl) | 2,532 | NOASSERTION | CUDA Core Compute Libraries (Thrust/CUB/cuSPARSE...). Shows up in kernel work. |
| [microsoft/pai](https://github.com/microsoft/pai) | 2,698 | MIT | Open-source cluster manager for AI from Microsoft Research. Useful prior art for schedulers. |

**Non-repo hardware you must know:** NVIDIA (H100/H200, B200/B300, GB200/GB300 NVL72, Rubin/VR NVL72, L40S, RTX PRO 6000), AMD (MI300X/MI325X, MI350X/MI355X, MI400-series), Google TPU (v5e/v5p, v6e Trillium, v7 Ironwood), AWS Trainium 2/3 (Inferentia is discontinued — AWS converged on Trainium), Intel Gaudi, Huawei Ascend, Apple Silicon (MLX/ANE), and the inference-only startups (Groq, Cerebras, SambaNova, Tenstorrent). **Datasheets > benchmarks > vibes** — and always ask format, dense/sparse, spec-vs-usable, and one-way-vs-bidirectional before quoting a number. See [HARDWARE.md](HARDWARE.md#2-accelerators-approximate-dense-verify).

---

## L1 · Runtime, Kernels & Compilers

**Decision notes.** PyTorch is the substrate. `torch.compile` + Triton get you 80% of achievable performance with 5% of the effort; hand-written CUDA is for the last 20% and for libraries (attention, GEMM, communication). Learn the *roofline* first, or you'll optimize the wrong kernel.

| Project | Stars | License | Notes |
|---|---:|---|---|
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | 104,296 | NOASSERTION | The default framework. Learn `torch.compile`, AMP, the caching allocator, and `torch.profiler`. |
| [triton-lang/triton](https://github.com/triton-lang/triton) | 20,347 | MIT | Python-like DSL for GPU kernels. The single highest-ROI kernel skill today. |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | 25,115 | BSD-3-Clause | Fast exact attention. Know why it exists (SRAM tiling) even if you never edit it. |
| [tile-ai/tilelang](https://github.com/tile-ai/tilelang) | 8,568 | NOASSERTION | Tile-based DSL for high-performance kernels across GPU/CPU/accelerators. Rising fast. |
| [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | 8,929 | MIT | Clean FP8 GEMM library from DeepSeek. Study it for FP8 numerics. |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | 10,557 | NOASSERTION | CUDA Templates + Python DSLs for GEMM. The reference for serious kernel authors. |
| [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens) | 3,748 | MIT | Tile primitives that make fast kernels easier to write. Great learning codebase. |
| [NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine) | 3,573 | Apache-2.0 | FP8/FP4 training and inference for transformers. Where the numerics live. |
| [NVIDIA/nccl](https://github.com/NVIDIA/nccl) | 5,151 | NOASSERTION | Multi-GPU collectives. Read `nccl-tests` output; memorize the algorithms. |
| [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests) | 1,673 | BSD-3-Clause | Benchmark the fabric. Do this before blaming the model. |
| [openucx/ucx](https://github.com/openucx/ucx) | 1,723 | NOASSERTION | Unified Communication X, the transport under MPI/NCCL/NIXL. |
| [ai-dynamo/nixl](https://github.com/ai-dynamo/nixl) | 1,296 | NOASSERTION | Inference Xfer Library: high-speed KV-cache movement. Core of disaggregated serving. |
| [openxla/xla](https://github.com/openxla/xla) | 4,582 | Apache-2.0 | The compiler behind JAX/TPU. Learn it when you hit TPU or `jax.jit` limits. |
| [apache/tvm](https://github.com/apache/tvm) | 13,822 | Apache-2.0 | ML compiler framework: graph + operator optimizations. Older but influential. |
| [modular/modular](https://github.com/modular/modular) | 29,938 | NOASSERTION | MAX + Mojo. Interesting portability play; watch, don't bet the farm yet. |
| [onnx/onnx](https://github.com/onnx/onnx) | 21,575 | Apache-2.0 | Interchange format. Needed the moment you leave Python. |
| [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | 22,049 | MIT | Cross-platform inference runtime. Strong for CPU/edge and non-NVIDIA. |
| [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino) | 10,981 | Apache-2.0 | Intel CPU/GPU/NPU optimization and deployment toolkit. |
| [ml-explore/mlx](https://github.com/ml-explore/mlx) | 28,722 | MIT | Apple-silicon array framework. The right tool for Mac inference/training. |
| [pytorch/executorch](https://github.com/pytorch/executorch) | 5,088 | NOASSERTION | PyTorch on mobile/embedded/edge. |
| [Dao-AILab/quack](https://github.com/Dao-AILab/quack) | 1,167 | Apache-2.0 | CuTe kernel collection; advanced kernel-engineering reference. |

---

## L2 · Training & Post-Training

**Decision notes.** Three separate problems hide here: **(1)** pretraining at scale (Megatron/DeepSpeed/Composer + a parallelism plan), **(2)** parameter-efficient fine-tuning (PEFT/LoRA, Unsloth, LlamaFactory, Axolotl, torchtune), and **(3)** preference/RL post-training (TRL, verl, OpenRLHF). Choose by *scale and method*, not by GitHub stars. For a first fine-tune, use a high-level trainer; drop to the framework only when you need control.

| Project | Stars | License | Notes |
|---|---:|---|---|
| [huggingface/transformers](https://github.com/huggingface/transformers) | 167,502 | Apache-2.0 | The model-definition layer everything integrates with. |
| [huggingface/datasets](https://github.com/huggingface/datasets) | 22,049 | Apache-2.0 | Dataset loading/streaming. The default data interface. |
| [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | 11,166 | Apache-2.0 | Fast tokenization. Understand BPE once, then never think about it. |
| [huggingface/accelerate](https://github.com/huggingface/accelerate) | 9,912 | Apache-2.0 | Launch training on any device/parallelism with minimal code change. |
| [huggingface/peft](https://github.com/huggingface/peft) | 21,781 | Apache-2.0 | LoRA/QLoRA/DoRA and friends: the reference implementation. |
| [huggingface/trl](https://github.com/huggingface/trl) | 19,489 | Apache-2.0 | SFT/DPO/GRPO/PPO trainers. Start here for post-training. |
| [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | 43,218 | Apache-2.0 | ZeRO stages, offload, inference kernels. The accessible path to multi-GPU training. |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | 18,115 | NOASSERTION | Peak-MFU large-scale training; TP/PP/EP reference. Steep, worth it. |
| [pytorch/torchtitan](https://github.com/pytorch/torchtitan) | 5,797 | BSD-3-Clause | PyTorch-native platform for generative-model pretraining. The PyTorch-team-blessed path. |
| [hpcaitech/ColossalAI](https://github.com/hpcaitech/ColossalAI) | 41,437 | Apache-2.0 | Unified parallelism (Gemini, 3D, MoE). Good if DeepSpeed/Megatron don't fit. |
| [Lightning-AI/pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning) | 31,395 | Apache-2.0 | Removes boilerplate; strong multi-GPU ergonomics. |
| [jax-ml/jax](https://github.com/jax-ml/jax) | 36,405 | Apache-2.0 | Composable transforms, `shard_map`, SPMD. The TPU/Google path. |
| [google/flax](https://github.com/google/flax) | 7,341 | Apache-2.0 | Neural-net library for JAX. |
| [keras-team/keras](https://github.com/keras-team/keras) | 64,365 | Apache-2.0 | Multi-backend (JAX/TF/PyTorch). Good for pedagogy and small scale. |
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | 77,759 | Apache-2.0 | Fastest path to a working LoRA/QLoRA fine-tune; big memory savings. |
| [hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory) | 75,409 | Apache-2.0 | Unified fine-tuning for 100+ LLMs/VLMs, config-driven. Great for teams. |
| [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) | 12,556 | Apache-2.0 | YAML-configured fine-tuning; popular for reproducible SFT/DPO recipes. |
| [meta-pytorch/torchtune](https://github.com/meta-pytorch/torchtune) | 5,811 | BSD-3-Clause | PyTorch-native post-training library. Clean, hackable. |
| [modelscope/ms-swift](https://github.com/modelscope/ms-swift) | 15,810 | Apache-2.0 | CPT/SFT/DPO/GRPO for 600+ LLMs and 300+ MLLMs. Broad model coverage. |
| [InternLM/xtuner](https://github.com/InternLM/xtuner) | 5,206 | Apache-2.0 | Training engine tuned for ultra-large MoE models. |
| [verl-project/verl](https://github.com/verl-project/verl) | 23,825 | Apache-2.0 | Flexible RL post-training (PPO/DAPO/GRPO): the framework behind many open reasoning models. |
| [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 10,080 | Apache-2.0 | Ray-based agentic RL framework, easy to scale. |
| [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | 6,652 | BSD-2-Clause | Triton kernels that cut training memory and raise throughput. Free wins. |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | 63,708 | MIT | Minimal GPT training. The best "learn by reading" codebase. Stable and complete rather than actively developed. |
| [karpathy/llm.c](https://github.com/karpathy/llm.c) | 31,134 | MIT | GPT-2 training in raw C/CUDA. Read it to see what PyTorch hides. A finished teaching artifact, not a maintained framework. |
| [NVIDIA-NeMo/NeMo](https://github.com/NVIDIA-NeMo/NeMo) | 18,571 | Apache-2.0 | Full-stack framework for LLM/multimodal/speech training and customization. |
| [allenai/open-instruct](https://github.com/allenai/open-instruct) | 3,895 | Apache-2.0 | AllenAI's production post-training codebase (the code behind Tülu 3). Good SFT/DPO/RLVR hygiene to copy. |

---

## L2b · RL Post-Training Infrastructure

**Why this gets its own layer.** RL post-training is no longer a fine-tuning footnote; it is where a large share of frontier compute now goes, and it has a *different* systems profile from pretraining. Pretraining is one tightly-coupled collective job. RL is two loosely-coupled workloads — a **rollout engine** (memory-bandwidth-bound generation) and a **trainer** (compute-bound backprop) — plus a **verifier/sandbox** tier, all feeding each other in a loop. Most of the difficulty is in the seams, not the math.

**Decision notes.** Pick by *scale tier first*: **single GPU / learning** → TRL or ART. **8-64 GPUs** → verl or rLLM. **100B+ and MoE** → slime, Miles, NeMo RL, or prime-rl. The framework matters less than the three things you must measure: rollout time, training time, and **weight-sync (reshard) time**. See [cheatsheets/rl-post-training.md](cheatsheets/rl-post-training.md) for the math and the systems invariants.

| Project | Stars | License | Notes |
|---|---:|---|---|
| [verl-project/verl](https://github.com/verl-project/verl) | 23,825 | Apache-2.0 | **The ecosystem default.** Hybrid-controller RL (FSDP *or* Megatron training × vLLM *or* SGLang rollout). Start here unless you have a reason not to. Migrated out of `volcengine/` in 2026 — use this slug. |
| [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 10,080 | Apache-2.0 | The classic Ray + vLLM **disaggregated** baseline. Best used to *learn* the architecture; no longer the throughput leader at frontier scale. |
| [OpenPipe/ART](https://github.com/OpenPipe/ART) | 10,792 | Apache-2.0 | Agent Reinforcement Trainer: GRPO for multi-step agents with on-the-job training. The most accessible agentic-RL entry point. |
| [THUDM/slime](https://github.com/THUDM/slime) | 8,632 | Apache-2.0 | Megatron + SGLang RL with one unified train/rollout/data path. Small enough to read end to end; behind the GLM releases. |
| [rllm-org/rllm](https://github.com/rllm-org/rllm) | 5,869 | Apache-2.0 | Agent RL with a solver-judge workflow and programmable reward functions. Good for multi-turn tasks. |
| [areal-project/AReaL](https://github.com/areal-project/AReaL) | 5,829 | Apache-2.0 | Async RL with a clean bridge for agent applications. Note the 2026 org move to `areal-project/`. |
| [PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers) | 4,696 | MIT | RL environments + evals library, backing the Prime Intellect Environments Hub. One of the two competing environment standards. |
| [huggingface/OpenEnv](https://github.com/huggingface/OpenEnv) | 2,687 | BSD-3-Clause | Gymnasium-style (`reset`/`step`/`state`) isolated Docker environments for agentic RL; adapters for TRL, Unsloth, SkyRL, ART, and more. The other standard. Young but moving fast. |
| [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL) | 2,401 | Apache-2.0 | Modular full-stack RL library with an explicit agent layer (`skyrl-agent` is a reference multi-turn implementation). |
| [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) | 2,152 | Apache-2.0 | Fully asynchronous RL, explicitly targeting 1T+ MoE on 1000+ GPUs. Pick it when asynchrony *is* the problem. |
| [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) | 2,058 | Apache-2.0 | NVIDIA's scalable RL toolkit with reproducible recipes (trained Nemotron-3). Vendor-fast-moving; day-0 model support. |
| [radixark/miles](https://github.com/radixark/miles) | 3,088 | Apache-2.0 | Enterprise fork co-evolving with slime; LLM + VLM post-training. |
| [verl-project/verl-recipe](https://github.com/verl-project/verl-recipe) | 338 | Apache-2.0 | End-to-end reproducible recipes (DAPO, ReTool). The fastest way to a *known-good* run. |
| [verl-project/rl-insight](https://github.com/verl-project/rl-insight) | 87 | Apache-2.0 | Online observability for RL: separates generation / training / weight-sync time across a distributed rollout. **Low stars, real gap** — you cannot debug RL throughput without this split. |
| [OpenHands/OpenHands](https://github.com/OpenHands/OpenHands) | 90,566 | MIT | Agent platform widely used as a rollout harness and sandbox target for agentic RL. |

**Not a framework, but part of the loop:** verifiable-reward tasks need a **benchmark with a programmatic checker** — [SWE-bench](https://github.com/SWE-bench/SWE-bench), [terminal-bench](https://github.com/harbor-framework/terminal-bench-1), or your own. The verifier is a service on the critical path, and its sandbox is a security boundary.

---

## L3 · Inference & Serving

**Decision notes.** Pick by *deployment target*, not popularity: **server GPU** → vLLM (default) or SGLang (prefix-heavy/structured); **max NVIDIA perf** → TensorRT-LLM; **CPU/edge/Mac** → llama.cpp; **one-command local** → Ollama; **Kubernetes fleet** → KServe + llm-d/Dynamo. Then optimize the *cost at your SLO*: quantization, prefix caching, and disaggregation usually beat micro-tuning.

| Project | Stars | License | Notes |
|---|---:|---|---|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 93,557 | Apache-2.0 | The default LLM server. PagedAttention, continuous batching, wide model/hardware support. |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | 36,985 | Apache-2.0 | RadixAttention prefix caching + structured output. Often wins on shared-prefix workloads. |
| [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | 14,792 | NOASSERTION | Highest NVIDIA throughput after engineering effort. Build step is real. |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | 130,811 | MIT | Runs everywhere; GGUF quantization; the "does it fit" baseline. |
| [ollama/ollama](https://github.com/ollama/ollama) | 182,731 | MIT | One-command local models. Great for dev, not for serving at scale. |
| [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) | 10,880 | Apache-2.0 | **Archived in 2026.** Historically influential, but use vLLM or SGLang for new deployments. |
| [InternLM/lmdeploy](https://github.com/InternLM/lmdeploy) | 8,103 | Apache-2.0 | Efficient serving with strong Chinese-model and edge coverage. |
| [microsoft/BitNet](https://github.com/microsoft/BitNet) | 40,652 | MIT | 1-bit/1.58-bit inference. Watch the accuracy trade carefully. |
| [turboderp-org/exllamav2](https://github.com/turboderp-org/exllamav2) | 4,632 | MIT | Fast local inference for consumer GPUs with EXL2 quantization. |
| [mudler/LocalAI](https://github.com/mudler/LocalAI) | 49,469 | MIT | OpenAI-compatible local stack (LLM, image, audio, TTS). The self-hosted drop-in. |
| [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm) | 23,230 | Apache-2.0 | Compile-and-deploy across GPUs, mobile, browsers, WebGPU. |
| [Tiiny-AI/PowerInfer](https://github.com/Tiiny-AI/PowerInfer) | 9,819 | MIT | Sparse activation on consumer GPUs. Nice research-to-practice case. |
| [triton-inference-server/server](https://github.com/triton-inference-server/server) | 11,067 | BSD-3-Clause | Multi-framework model server with ensembles and dynamic batching. Still the enterprise workhorse. |
| [bentoml/BentoML](https://github.com/bentoml/BentoML) | 8,897 | Apache-2.0 | Build/serve AI apps and inference APIs; good ergonomics for custom pipelines. |
| [bentoml/OpenLLM](https://github.com/bentoml/OpenLLM) | 12,554 | Apache-2.0 | Run any open LLM as an OpenAI-compatible API. |
| [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) | 3,864 | Apache-2.0 | Production quantization (FP8, INT8, INT4, W4A16) built for vLLM. |
| [AutoGPTQ/AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) | 5,066 | MIT | **Archived.** The maintained successor is [GPTQModel](https://github.com/ModelCloud/GPTQModel). |
| [casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ) | 2,346 | MIT | **Archived.** For new work use [llm-compressor](https://github.com/vllm-project/llm-compressor) or GPTQModel. |
| [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | 8,518 | MIT | 8-bit/4-bit primitives used across the fine-tuning ecosystem. |
| [nunchux-ai/nunchaku](https://github.com/nunchux-ai/nunchaku) | 3,959 | Apache-2.0 | SVDQuant 4-bit diffusion inference. |
| [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) | 3,966 | Apache-2.0 | Quantized attention with big speedups. Drop-in wins for long context. |
| [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) | 8,257 | NOASSERTION | Datacenter-scale disaggregated inference (NVIDIA). |
| [llm-d/llm-d](https://github.com/llm-d/llm-d) | 4,790 | Apache-2.0 | K8s-native distributed inference: KV-aware routing, disaggregation. |
| [vllm-project/aibrix](https://github.com/vllm-project/aibrix) | 5,129 | Apache-2.0 | Pluggable inference infra components (autoscaler, cache, router). |
| [kserve/kserve](https://github.com/kserve/kserve) | 6,101 | Apache-2.0 | Standardized multi-framework serving on Kubernetes, incl. LLM inference. |
| [gpustack/gpustack](https://github.com/gpustack/gpustack) | 5,812 | Apache-2.0 | GPU cluster manager for serving vLLM/SGLang + on-demand GPU dev boxes. |
| [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router) | 6,080 | Apache-2.0 | Route requests to models by intent or complexity; send easy queries to cheap models. |
| [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) | 2,949 | Apache-2.0 | vLLM on Huawei Ascend NPUs. |

---

## L4 · Data & Storage

**Decision notes.** Training and RAG both die on data I/O, not on GPUs. Three rules: **(1)** never store millions of tiny files for training; use sharded formats (WebDataset, Mosaic Streaming, Lance); **(2)** separate object storage (cheap, durable) from a caching/parallel FS (fast); Fluid/JuiceFS/Alluxio exist for this; **(3)** version your data as seriously as your code (DVC/lakeFS) and track it (lineage).

| Project | Stars | License | Notes |
|---|---:|---|---|
| [minio/minio](https://github.com/minio/minio) | 61,328 | AGPL-3.0 | **Archived in 2026.** Widely deployed, but plan a migration; see SeaweedFS and RustFS below. |
| [seaweedfs/seaweedfs](https://github.com/seaweedfs/seaweedfs) | 35,390 | Apache-2.0 | Distributed object store (S3), file system, and blob store. The most established MinIO alternative. |
| [rustfs/rustfs](https://github.com/rustfs/rustfs) | 34,712 | Apache-2.0 | Rust S3-compatible object storage. Fast-moving alternative for new on-prem deployments. |
| [ceph/ceph](https://github.com/ceph/ceph) | 17,107 | NOASSERTION | Distributed object/block/file storage. Heavy but battle-tested. |
| [juicedata/juicefs](https://github.com/juicedata/juicefs) | 14,525 | Apache-2.0 | POSIX FS on top of object storage + Redis. Great for shared training data. |
| [Alluxio/alluxio](https://github.com/Alluxio/alluxio) | 7,247 | Apache-2.0 | Data orchestration/caching between storage and compute. |
| [fluid-cloudnative/fluid](https://github.com/fluid-cloudnative/fluid) | 1,986 | Apache-2.0 | Elastic dataset abstraction + caching on K8s. |
| [apache/spark](https://github.com/apache/spark) | 44,165 | Apache-2.0 | Batch/stream processing; the incumbent for tabular data at scale. |
| [apache/flink](https://github.com/apache/flink) | 26,398 | Apache-2.0 | True streaming with state. Use when latency matters. |
| [apache/kafka](https://github.com/apache/kafka) | 33,933 | Apache-2.0 | Event streaming backbone. The default for event-driven AI. |
| [redpanda-data/redpanda](https://github.com/redpanda-data/redpanda) | 12,602 | NOASSERTION | Kafka-API-compatible, no JVM. Lower ops overhead. |
| [duckdb/duckdb](https://github.com/duckdb/duckdb) | 42,095 | MIT | In-process analytics SQL. The fastest way to explore a dataset. |
| [pola-rs/polars](https://github.com/pola-rs/polars) | 40,097 | MIT | Rust DataFrame engine. Pandas for people who care about speed. |
| [apache/iceberg](https://github.com/apache/iceberg) | 9,321 | Apache-2.0 | Open table format for lakehouses. The 2026 default for big tabular data. |
| [delta-io/delta](https://github.com/delta-io/delta) | 9,040 | Apache-2.0 | Lakehouse table format (Databricks lineage). |
| [apache/hudi](https://github.com/apache/hudi) | 6,294 | Apache-2.0 | Upserts and incremental processing on the lake. |
| [lance-format/lance](https://github.com/lance-format/lance) | 7,141 | Apache-2.0 | Multimodal lakehouse format: fast random access + vectors. Excellent for AI data. |
| [treeverse/dvc](https://github.com/treeverse/dvc) | 15,911 | Apache-2.0 | Data version control. Git for datasets and models. |
| [treeverse/lakeFS](https://github.com/treeverse/lakeFS) | 5,554 | NOASSERTION | Git-like branching for the data lake. |
| [feast-dev/feast](https://github.com/feast-dev/feast) | 7,323 | Apache-2.0 | Feature store. The train/serve skew killer. |
| [HumanSignal/label-studio](https://github.com/HumanSignal/label-studio) | 28,442 | Apache-2.0 | Multi-type labeling/annotation. The industry default. |
| [argilla-io/argilla](https://github.com/argilla-io/argilla) | 5,147 | Apache-2.0 | Human-in-the-loop data curation for LLMs. |
| [Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured) | 15,558 | Apache-2.0 | Document ETL into structured data. |
| [docling-project/docling](https://github.com/docling-project/docling) | 68,672 | MIT | Best-in-class document → structured markdown for RAG. |
| [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | 90,917 | Apache-2.0 | OCR for PDFs/images in 100+ languages. |
| [opendataloader-project/opendataloader-pdf](https://github.com/opendataloader-project/opendataloader-pdf) | 29,572 | Apache-2.0 | PDF → AI-ready data with layout awareness. |

**Also worth knowing:** [MosaicML Streaming](https://github.com/mosaicml/streaming) and [WebDataset](https://github.com/webdataset/webdataset) (sharded training formats), [Daft](https://github.com/Eventual-Inc/Daft) and [Ray Data](https://github.com/ray-project/ray) (distributed multimodal preprocessing), [LitData](https://github.com/Lightning-AI/litdata), [Petastorm](https://github.com/uber/petastorm).

---

## L5 · Orchestration & Scheduling

**Decision notes.** Three families: **K8s** (fleet, multi-tenant serving, platform), **Slurm** (HPC-style training clusters, what labs actually run), **Ray** (Python-native distributed compute for both). Add **SkyPilot** to place work across clouds/regions and **Airflow/Prefect/Dagster** for orchestration of *pipelines* (not GPUs). Don't run both Slurm and K8s unless you have a reason you can articulate.

| Project | Stars | License | Notes |
|---|---:|---|---|
| [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes) | 128,267 | Apache-2.0 | The substrate for GPU platforms. Learn Jobs, quotas, and scheduling semantics. |
| [ray-project/ray](https://github.com/ray-project/ray) | 44,005 | Apache-2.0 | Distributed Python runtime + Train/Tune/Serve/Data. Excellent for pipelines that mix CPU and GPU. |
| [SchedMD/slurm](https://github.com/SchedMD/slurm) | 4,427 | NOASSERTION | HPC workload manager. Topology-aware, gang-scheduling by default. |
| [skypilot-org/skypilot](https://github.com/skypilot-org/skypilot) | 10,694 | Apache-2.0 | Run a job on any cloud/region/K8s/Slurm with one YAML. Huge cost lever. |
| [kubernetes-sigs/kueue](https://github.com/kubernetes-sigs/kueue) | 3,051 | Apache-2.0 | K8s-native job queueing, quotas, and fair sharing for batch/AI. |
| [volcano-sh/volcano](https://github.com/volcano-sh/volcano) | 6,006 | Apache-2.0 | CNCF batch system: gang scheduling, plugins, queues. |
| [kubernetes-sigs/lws](https://github.com/kubernetes-sigs/lws) | 871 | Apache-2.0 | LeaderWorkerSet: deploy a group of pods as one unit (multi-host inference). |
| [kubernetes-sigs/jobset](https://github.com/kubernetes-sigs/jobset) | 346 | Apache-2.0 | JobSet: K8s-native API for distributed training/HPC jobs. |
| [ray-project/kuberay](https://github.com/ray-project/kuberay) | 2,720 | Apache-2.0 | Run Ray clusters on Kubernetes, properly. |
| [kubeflow/trainer](https://github.com/kubeflow/trainer) | 2,242 | Apache-2.0 | Distributed training and fine-tuning on K8s. |
| [kubeflow/kubeflow](https://github.com/kubeflow/kubeflow) | 15,899 | Apache-2.0 | The original ML platform on K8s. Big; pick components, not the whole thing. |
| [kubeflow/pipelines](https://github.com/kubeflow/pipelines) | 4,238 | Apache-2.0 | ML pipelines (Argo-backed) with lineage/artifacts. |
| [argo-workflows](https://github.com/argoproj/argo-workflows) | 17,032 | Apache-2.0 | Container-native workflow engine for K8s. |
| [argoproj/argo-cd](https://github.com/argoproj/argo-cd) | 24,364 | Apache-2.0 | GitOps continuous delivery. How you actually change a platform. |
| [apache/airflow](https://github.com/apache/airflow) | 47,145 | Apache-2.0 | Batch DAG orchestration. The incumbent for data pipelines. |
| [PrefectHQ/prefect](https://github.com/PrefectHQ/prefect) | 24,002 | Apache-2.0 | Pythonic orchestration with better DX than Airflow. |
| [dagster-io/dagster](https://github.com/dagster-io/dagster) | 16,264 | Apache-2.0 | Asset-centric orchestration; strong data-awareness. |
| [flyteorg/flyte](https://github.com/flyteorg/flyte) | 7,677 | Apache-2.0 | Type-safe distributed orchestration on K8s, strong for ML. |
| [Netflix/metaflow](https://github.com/Netflix/metaflow) | 10,300 | Apache-2.0 | Human-friendly ML/data workflows; great for experiments. |
| [zenml-io/zenml](https://github.com/zenml-io/zenml) | 5,611 | Apache-2.0 | MLOps framework that stitches your existing tools into pipelines. |
| [dstackai/dstack](https://github.com/dstackai/dstack) | 2,277 | MPL-2.0 | Unified orchestration across heterogeneous AI compute, dev-friendly. |
| [facebookincubator/submitit](https://github.com/facebookincubator/submitit) | 1,642 | MIT | Submit Python jobs to Slurm from a laptop. Small and brilliant. |
| [helm/helm](https://github.com/helm/helm) | 30,308 | Apache-2.0 | K8s package manager. |
| [hashicorp/terraform](https://github.com/hashicorp/terraform) | 49,864 | NOASSERTION | Infrastructure as code. Note the BUSL license change. |

---

## L6 · Retrieval, Context & Memory

**Decision notes.** Order of impact: **chunking/parsing quality > hybrid search > reranking > embedding model > ANN index choice.** Most "vector DB" problems are actually document-processing problems. Start with `pgvector` if you already run Postgres; go dedicated (Qdrant/Milvus/Weaviate) past ~10M vectors or when you need filtering at scale. Add GraphRAG only when relationships genuinely matter.

| Project | Stars | License | Notes |
|---|---:|---|---|
| [facebookresearch/faiss](https://github.com/facebookresearch/faiss) | 41,129 | MIT | The ANN library. Not a database; a building block. |
| [milvus-io/milvus](https://github.com/milvus-io/milvus) | 46,349 | Apache-2.0 | Cloud-native vector DB built for very large scale. |
| [qdrant/qdrant](https://github.com/qdrant/qdrant) | 35,027 | Apache-2.0 | Excellent DX, filtering, quantization. Great default dedicated choice. |
| [weaviate/weaviate](https://github.com/weaviate/weaviate) | 16,875 | NOASSERTION | Vectors + objects + hybrid search, built-in modules. |
| [chroma-core/chroma](https://github.com/chroma-core/chroma) | 29,477 | Apache-2.0 | Easiest to start with; great for prototypes. |
| [pgvector/pgvector](https://github.com/pgvector/pgvector) | 23,300 | NOASSERTION | Vectors inside Postgres. Often the correct answer. |
| [lancedb/lancedb](https://github.com/lancedb/lancedb) | 11,637 | Apache-2.0 | Embedded multimodal vector+columnar store. Pairs with Lance. |
| [opensearch-project/OpenSearch](https://github.com/opensearch-project/OpenSearch) | 13,843 | Apache-2.0 | Full-text + vector hybrid at scale; the AWS-backed fork. |
| [elastic/elasticsearch](https://github.com/elastic/elasticsearch) | 78,226 | NOASSERTION | The incumbent search engine; hybrid search is mature. |
| [meilisearch/meilisearch](https://github.com/meilisearch/meilisearch) | 59,541 | NOASSERTION | Fast, ergonomic search API for product-facing search. |
| [typesense/typesense](https://github.com/typesense/typesense) | 26,851 | GPL-3.0 | Low-ops search alternative to Algolia. |
| [vespa-engine/vespa](https://github.com/vespa-engine/vespa) | 7,123 | Apache-2.0 | Industrial-strength ranking + vectors. Steeper, very powerful. |
| [huggingface/sentence-transformers](https://github.com/huggingface/sentence-transformers) | 19,160 | Apache-2.0 | Embeddings + reranking. The default embedding toolkit. |
| [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) | 12,231 | MIT | BGE embeddings and rerankers; strong quality for the cost. |
| [neuml/txtai](https://github.com/neuml/txtai) | 13,006 | Apache-2.0 | All-in-one embeddings database + pipelines. |
| [run-llama/llama_index](https://github.com/run-llama/llama_index) | 52,463 | MIT | Data framework for RAG/agents; the largest set of connectors and index types. |
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | 147,608 | MIT | Broad ecosystem glue. Use targeted pieces; avoid unbounded abstractions. |
| [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | 91,990 | Apache-2.0 | End-to-end RAG engine with strong parsing (deep document understanding). |
| [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) | 40,063 | MIT | Simple, fast GraphRAG. Much cheaper than Microsoft's GraphRAG. |
| [microsoft/graphrag](https://github.com/microsoft/graphrag) | 36,290 | MIT | Graph-based RAG; powerful, expensive to index. |
| [getzep/graphiti](https://github.com/getzep/graphiti) | 31,637 | Apache-2.0 | Real-time temporally-aware knowledge graphs for agents. |
| [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) | 39,050 | MIT | Reasoning-based retrieval over long docs without a vector store. |
| [topoteretes/cognee](https://github.com/topoteretes/cognee) | 32,000 | Apache-2.0 | Memory/graph layer with pipelines for RAG over your data. |

---

## L7 · Application & Agent Runtime

**Decision notes.** You need four things to run agents in production: **a gateway** (keys, routing, budgets, fallbacks), **a durable execution model** (retries, long-running state), **sandboxing** (untrusted generated code), and **observability** (L8). Frameworks come after those. Choose a framework by how well it lets you *debug*, not by how it demos.

| Project | Stars | License | Notes |
|---|---:|---|---|
| [BerriAI/litellm](https://github.com/BerriAI/litellm) | 60,952 | NOASSERTION | The default AI gateway: 100+ providers, OpenAI-compatible, budgets, fallbacks, cost tracking. |
| [Portkey-AI/gateway](https://github.com/Portkey-AI/gateway) | 13,164 | MIT | Fast gateway with guardrails and routing. |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | 43,058 | MIT | Durable, stateful agent graphs with checkpointing and human-in-the-loop. The production default. |
| [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | 59,560 | MIT | Role-based multi-agent teams. Fast to demo. |
| [microsoft/autogen](https://github.com/microsoft/autogen) | 61,350 | CC-BY-4.0 | Multi-agent conversations and orchestration research. |
| [agno-agi/agno](https://github.com/agno-agi/agno) | 42,663 | Apache-2.0 | Full platform for building/running/managing agents. |
| [pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai) | 20,546 | MIT | Typed agents with validation-first design. |
| [huggingface/smolagents](https://github.com/huggingface/smolagents) | 29,781 | Apache-2.0 | Minimal agents that write code as actions. |
| [openai/openai-agents-python](https://github.com/openai/openai-agents-python) | 29,963 | MIT | Lightweight multi-agent workflows, provider-agnostic. |
| [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) | 38,592 | MIT | Program,don't prompt. Optimize prompts/pipelines with data. |
| [temporalio/temporal](https://github.com/temporalio/temporal) | 23,602 | MIT | Durable execution via a polling worker + external database. The most battle-tested answer for long-running agent workflows. |
| [restatedev/restate](https://github.com/restatedev/restate) | 4,543 | NOASSERTION | Durable execution as a self-contained push-based runtime. Lower ops burden than Temporal when you do not want a separate DB cluster. |
| [dbos-inc/dbos-transact-py](https://github.com/dbos-inc/dbos-transact-py) | 1,619 | MIT | Durable execution **embedded in your existing Postgres**. Pick it when you already have Postgres and want no new infrastructure. |
| [inngest/inngest](https://github.com/inngest/inngest) | 5,932 | NOASSERTION | Event-driven durable workflows with good DX; strong for step-function-style agent pipelines. |
| [modelcontextprotocol/modelcontextprotocol](https://github.com/modelcontextprotocol/modelcontextprotocol) | 9,427 | NOASSERTION | The MCP spec. **Current revision: 2026-07-28, which is stateless** — the old `initialize` handshake and session IDs are gone, and each request now carries its protocol version in `_meta`. Most tutorials still describe the pre-2026 lifecycle; check the spec revision before trusting a guide. |
| [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) | 91,137 | NOASSERTION | Reference MCP servers. Read them before writing your own. |
| [a2aproject/A2A](https://github.com/a2aproject/A2A) | 26,103 | Apache-2.0 | Agent-to-agent protocol for cross-vendor interop. |
| [ag-ui-protocol/ag-ui](https://github.com/ag-ui-protocol/ag-ui) | 16,453 | MIT | Agent↔frontend streaming protocol. |
| [e2b-dev/E2B](https://github.com/e2b-dev/E2B) | 14,275 | Apache-2.0 | Secure cloud sandboxes for agent-generated code. Managed, fast to start. |
| [kubernetes-sigs/agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) | 4,219 | Apache-2.0 | K8s CRD for isolated, stateful agent sandboxes. **The default choice if you already run K8s** and want sandboxes as a platform primitive. |
| [google/gvisor](https://github.com/google/gvisor) | 19,636 | Apache-2.0 | User-space kernel; the container isolation layer under many sandboxing products. |
| [firecracker-microvm/firecracker](https://github.com/firecracker-microvm/firecracker) | 37,283 | Apache-2.0 | microVM monitor. The strong-isolation primitive: separate guest kernel, ~125 ms boot. |
| [daytonaio/daytona](https://github.com/daytonaio/daytona) | 71,569 | NOASSERTION | **Archived (frozen mid-2026; core development moved private).** High star count is historical. Use E2B, microsandbox, or agent-sandbox instead. |
| [superradcompany/microsandbox](https://github.com/superradcompany/microsandbox) | 8,646 | Apache-2.0 | Local-first microVM runtime for running untrusted agent code. |
| [browser-use/browser-use](https://github.com/browser-use/browser-use) | 117,621 | MIT | Browser automation for agents. |
| [firecrawl/firecrawl](https://github.com/firecrawl/firecrawl) | 190,348 | AGPL-3.0 | Web → LLM-ready markdown at scale. AGPL, so check your use. |
| [n8n-io/n8n](https://github.com/n8n-io/n8n) | 206,979 | NOASSERTION | Visual workflow automation with AI nodes. Fair-code, not OSI. |
| [langgenius/dify](https://github.com/langgenius/dify) | 158,137 | NOASSERTION | Agentic workflows + RAG in one self-hostable workspace. |
| [langflow-ai/langflow](https://github.com/langflow-ai/langflow) | 155,494 | MIT | Visual builder for agents/flows. |
| [open-webui/open-webui](https://github.com/open-webui/open-webui) | 154,283 | NOASSERTION | Self-hosted ChatGPT-like UI; connects to Ollama/OpenAI. |
| [lobehub/lobehub](https://github.com/lobehub/lobehub) | 83,133 | NOASSERTION | Polished multi-agent chat UI/workspace. |

---

## L8 · Observability, Evaluation, Safety & Governance

**Decision notes.** Three different jobs, often confused: **tracing** (what happened?), **evaluation** (is it good?), and **guardrails/red-teaming** (is it safe?). You need all three, and you need them *in CI*. An eval that only runs in a notebook is a hobby. Start with OpenTelemetry-compatible tracing so you're never locked in.

| Project | Stars | License | Notes |
|---|---:|---|---|
| [langfuse/langfuse](https://github.com/langfuse/langfuse) | 35,643 | NOASSERTION | Open-source LLM tracing, evals, prompts, datasets. The default. |
| [Arize-ai/phoenix](https://github.com/Arize-ai/phoenix) | 11,785 | NOASSERTION | Tracing + evals, strong for RAG and drift analysis. |
| [comet-ml/opik](https://github.com/comet-ml/opik) | 22,500 | Apache-2.0 | Tracing + evals + guardrails from Comet. |
| [traceloop/openllmetry](https://github.com/traceloop/openllmetry) | 7,481 | Apache-2.0 | OpenTelemetry instrumentation for LLM apps. Vendor-neutral tracing. |
| [mlflow/mlflow](https://github.com/mlflow/mlflow) | 28,342 | Apache-2.0 | Experiment tracking, registry, and now GenAI tracing/eval. The safe default. |
| [aimhubio/aim](https://github.com/aimhubio/aim) | 6,283 | Apache-2.0 | Fast, self-hosted experiment tracker with great UI. |
| [clearml/clearml](https://github.com/clearml/clearml) | 6,902 | Apache-2.0 | Full MLOps suite: experiments, data, pipelines, serving. |
| [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 14,186 | MIT | The canonical few-shot LM eval harness. Reproducible numbers. |
| [huggingface/lighteval](https://github.com/huggingface/lighteval) | 2,554 | MIT | All-in-one eval across backends; good CI ergonomics. |
| [open-compass/opencompass](https://github.com/open-compass/opencompass) | 7,507 | Apache-2.0 | Large-scale LLM evaluation platform, wide model coverage. |
| [stanford-crfm/helm](https://github.com/stanford-crfm/helm) | 2,939 | Apache-2.0 | Holistic evaluation across scenarios and metrics. |
| [UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) | 2,980 | MIT | Rigorous eval framework from the UK AI Safety Institute. |
| [promptfoo/promptfoo](https://github.com/promptfoo/promptfoo) | 25,892 | MIT | Test prompts/agents/RAG in CI; also red-teaming. |
| [confident-ai/deepeval](https://github.com/confident-ai/deepeval) | 18,741 | Apache-2.0 | LLM unit tests: hallucination, faithfulness, relevancy. |
| [vibrantlabsai/ragas](https://github.com/vibrantlabsai/ragas) | 15,991 | Apache-2.0 | RAG-specific metrics. Pair with a RAG test set. |
| [Giskard-AI/giskard-oss](https://github.com/Giskard-AI/giskard-oss) | 5,886 | Apache-2.0 | Testing + evaluation for LLM agents. |
| [evidentlyai/evidently](https://github.com/evidentlyai/evidently) | 7,982 | Apache-2.0 | Data/model/LLM drift and monitoring. |
| [mlcommons/inference](https://github.com/mlcommons/inference) | 1,644 | Apache-2.0 | MLPerf inference reference implementations. Benchmark methodology standard. |
| [NVIDIA/garak](https://github.com/NVIDIA/garak) | 9,523 | Apache-2.0 | LLM vulnerability scanner / red-teaming. Run it before someone else does. |
| [NVIDIA-NeMo/Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) | 7,274 | NOASSERTION | Programmable guardrails (rails, flows, dialog control) via Colang. |
| [guardrails-ai/guardrails](https://github.com/guardrails-ai/guardrails) | 7,506 | Apache-2.0 | Validation-style guardrails for structured outputs and safety. |
| [protectai/llm-guard](https://github.com/protectai/llm-guard) | 3,218 | MIT | **Archived.** Input/output scanners for injection, PII, and secrets; pair with Guardrails or NeMo Guardrails. |
| [meta-llama/PurpleLlama](https://github.com/meta-llama/PurpleLlama) | 4,425 | NOASSERTION | Meta's LLM security/eval toolset (incl. CyberSecEval). |
| [prometheus/prometheus](https://github.com/prometheus/prometheus) | 66,467 | Apache-2.0 | Metrics. Pair with DCGM exporter for GPU fleet truth. |
| [grafana/grafana](https://github.com/grafana/grafana) | 77,240 | AGPL-3.0 | Dashboards for GPU/utilization/cost. AGPL. |
| [open-telemetry/opentelemetry-collector](https://github.com/open-telemetry/opentelemetry-collector) | 7,649 | Apache-2.0 | Vendor-neutral telemetry pipeline. Instrument once, switch backends freely. |

---

## Appendix · Learning & reference repos

Not part of the runtime stack, but they *teach* it. See [RESOURCES.md](RESOURCES.md) for courses, papers, and communities.

| Project | Stars | License | Notes |
|---|---:|---|---|
| [stas00/ml-engineering](https://github.com/stas00/ml-engineering) | 19,446 | CC-BY-SA-4.0 | The best free book on the engineering/debugging side of ML. |
| [bojieli/ai-infra-book](https://github.com/bojieli/ai-infra-book) | 6,216 | Apache-2.0 | Quantitative derivation of inference/training system design (zh, with tooling). |
| [HuaizhengZhang/AI-Infra-from-Zero-to-Hero](https://github.com/HuaizhengZhang/AI-Infra-from-Zero-to-Hero) | 4,422 | MIT | Reading list of systems papers + industry practice. |
| [deepseek-ai/open-infra-index](https://github.com/deepseek-ai/open-infra-index) | 8,077 | CC0-1.0 | Production-tested infra tool index from a frontier lab. |
| [mryab/efficient-dl-systems](https://github.com/mryab/efficient-dl-systems) | 1,041 | MIT | Free course on efficient deep-learning systems. |
| [modular/llm-inference-handbook](https://github.com/modular/llm-inference-handbook) | 453 | Apache-2.0 | Concise inference fundamentals. |
| [liguodongiot/llm-action](https://github.com/liguodongiot/llm-action) | 25,152 | Apache-2.0 | Large practical LLM-engineering guide (zh/en). |
| [ai-infra-curriculum/ai-infra-engineer-learning](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning) | 1,774 | MIT | Structured curriculum for AI infra engineers. |
| [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | 106,351 | NOASSERTION | Build a GPT from zero. Pair with Stage 1. |
| [mlabonne/llm-course](https://github.com/mlabonne/llm-course) | 83,473 | Apache-2.0 | Roadmap + Colab notebooks for LLM engineering. |
| [microsoft/generative-ai-for-beginners](https://github.com/microsoft/generative-ai-for-beginners) | 121,180 | MIT | 21 lessons, very gentle on-ramp. |
| [GokuMohandas/Made-With-ML](https://github.com/GokuMohandas/Made-With-ML) | 49,722 | MIT | Production-grade ML systems, end to end. |
| [DataTalksClub/mlops-zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) | 15,395 | NOASSERTION | Free 9-week MLOps course. |
| [DataTalksClub/data-engineering-zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp) | 46,124 | NOASSERTION | Free data-engineering course. |
| [d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en) | 29,813 | NOASSERTION | Dive into Deep Learning, free interactive textbook. Content is stable; the repo itself is quiet. |
| [dair-ai/ML-YouTube-Courses](https://github.com/dair-ai/ML-YouTube-Courses) | 17,434 | CC0-1.0 | Index of free ML course videos. Frozen list, still useful as a starting point. |
| [NirDiamant/agents-towards-production](https://github.com/NirDiamant/agents-towards-production) | 21,633 | NOASSERTION | Production agent patterns, code-first. |

---

**See also:** [HARDWARE.md](HARDWARE.md) for the physics behind these choices, and [cheatsheets/](cheatsheets/) for the formulas.
