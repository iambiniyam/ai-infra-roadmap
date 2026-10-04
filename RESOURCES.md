# 📚 Resources

Everything here is **free to access** unless explicitly marked *(paid)*. Ordered by how much you get per hour invested.

---

## Courses (free, high signal)

| Course | Provider | Best for | Why it's here |
|---|---|---|---|
| [**CS336 — Language Modeling from Scratch**](https://stanford-cs336.github.io/) | Stanford | Stages 1–3 | The single best end-to-end LLM-systems course. Builds tokenizer → transformer → training → inference → scaling. Lectures on YouTube. |
| [**Deep Learning Systems**](https://dlsyscourse.org/) | CMU 10-414/714 | Stage 1–2 | Builds autograd, a compiler, and a runtime from scratch. Explains PyTorch. |
| [**Efficient Deep Learning Systems**](https://github.com/mryab/efficient-dl-systems) | Yandex/HSE | Stages 1–3 | Free materials on quantization, distributed training, and inference. Systems-first. |
| [**TinyML & Efficient Deep Learning**](https://hanlab.mit.edu/courses/2023-fall-65940) | MIT 6.5940 | Stage 5 | Pruning, quantization, NAS, efficient inference. |
| [**Parallel Computing (CS267)**](https://sites.google.com/lbl.gov/cs267-spr2023) | Berkeley | Stage 3 | The MPI/collective/topology foundation under NCCL. |
| [**Parallel Computer Architecture (15-418/618)**](https://www.cs.cmu.edu/~418/) | CMU | Stage 5 | Why GPUs are shaped the way they are. |
| [**Distributed Systems (6.5840/6.824)**](https://pdos.csail.mit.edu/6.824/) | MIT | Stage 4 | Consensus, replication, fault tolerance — you will need all three. |
| [**Database Systems (15-445)**](https://15445.courses.cs.cmu.edu/) | CMU | Stage 4 | Storage/indexing/query execution. Explains why your data layer is slow. |
| [**GPU MODE lectures**](https://github.com/gpu-mode/lectures) | Community | Stages 1–5 | CUDA, Triton, and kernel optimization from practitioners. Also a Discord. |
| [**Hugging Face LLM Course**](https://huggingface.co/learn/llm-course) | Hugging Face | Stage 1 | Transformers, datasets, fine-tuning, evaluation — practical and current. |
| [**Full Stack Deep Learning**](https://fullstackdeeplearning.com/course/) | FSDL | Stage 4 | The product/ops side of shipping ML. |
| [**MLOps Zoomcamp**](https://github.com/DataTalksClub/mlops-zoomcamp) | DataTalks.Club | Stage 4 | Free 9-week MLOps course with a real project. |
| [**Data Engineering Zoomcamp**](https://github.com/DataTalksClub/data-engineering-zoomcamp) | DataTalks.Club | Stage 4 | Pipelines, warehouses, orchestration — the L4 foundation. |
| [**MIT Missing Semester**](https://missing.csail.mit.edu/) | MIT | Stage 0 | Shell, git, debugging. Do this before anything else. |
| [**Performance Engineering (6.172)**](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/) | MIT OCW | Stage 0/5 | Caches, profiling, parallelism — durable fundamentals. |
| [**Made With ML**](https://github.com/GokuMohandas/Made-With-ML) | Goku Mohandas | Stage 4 | Design → develop → deploy → iterate, end to end. |
| [**Fast.ai**](https://course.fast.ai/) | fast.ai | Stage 1 | Top-down practical deep learning. |

---

## Books

**Free**

| Book | Why |
|---|---|
| [**Dive into Deep Learning**](https://d2l.ai/) | Interactive, multi-framework, math included |
| [**The Little Book of Deep Learning**](https://fleuret.org/public/lbdl.pdf) | The best short on-ramp to the concepts |
| [**Understanding Deep Learning**](https://udlbook.github.io/udlbook/) | Rigorous and readable, free PDF |
| [**Deep Learning**](https://www.deeplearningbook.org/) (Goodfellow et al.) | The reference; use it as a dictionary |
| [**Reinforcement Learning: An Introduction**](http://incompleteideas.net/book/the-book-2nd.html) (Sutton & Barto) | Needed for RLHF/RLVR |
| [**The Datacenter as a Computer**](https://research.google/pubs/the-datacenter-as-a-computer-an-introduction-to-the-design-of-warehouse-scale-machines/) | Warehouse-scale thinking |
| [**Google SRE Book**](https://sre.google/books/) | SLOs, error budgets, incidents |
| [**ML Engineering Open Book**](https://github.com/stas00/ml-engineering) | Hardware, debugging, SLURM — the practical companion to Stages 1–4 |

**Paid, and worth it**

| Book | Why |
|---|---|
| *Designing Machine Learning Systems* — Chip Huyen | The lifecycle, from a practitioner who's shipped it |
| *AI Engineering* — Chip Huyen | The 2025+ successor; evaluation and product reality |
| *Programming Massively Parallel Processors* — Kirk & Hwu | The CUDA textbook |
| *Computer Architecture: A Quantitative Approach* — Hennessy & Patterson | Where the bandwidth numbers come from |
| *Systems Performance* — Brendan Gregg | The USE method; read it once, use it forever |
| *High Performance Python* — Gorelick & Ozsvald | The Python-side performance work |

---

## Papers

Read in this order. You don't need every paper to start — but you should be able to *summarize* every paper below.

### Foundations

| Paper | Why it matters |
|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | The architecture everything descends from |
| [Scaling Laws for Neural LMs](https://arxiv.org/abs/2001.08361) | How loss scales with compute/data/params |
| [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) | The tokens-per-param heuristic |
| [GPT-3](https://arxiv.org/abs/2005.14165) | Few-shot learning and the modern training recipe |
| [Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | A rare, detailed account of a frontier training run |

### Distributed training

| Paper | Why it matters |
|---|---|
| [ZeRO: memory optimizations](https://arxiv.org/abs/1910.02054) | The sharding model under FSDP and DeepSpeed |
| [Megatron-LM](https://arxiv.org/abs/1909.08053) | Tensor parallelism, defined |
| [Efficient Large-Scale LM Training (3D parallelism)](https://arxiv.org/abs/2105.04663) | How TP/PP/DP compose |
| [GPipe](https://arxiv.org/abs/1811.06965) | Pipeline parallelism and bubbles |
| [Reducing Activation Recomputation (sequence parallel)](https://arxiv.org/abs/2205.05198) | Activation memory math |
| [ZeRO-Offload](https://arxiv.org/abs/2101.06840) / [ZeRO-Infinity](https://arxiv.org/abs/2104.07857) | Offloading to CPU/NVMe |
| [Ring Attention](https://arxiv.org/abs/2310.01889) | Long-context training across devices |

### Inference & serving

| Paper | Why it matters |
|---|---|
| [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | KV-cache paging; the basis of modern serving |
| [Orca: continuous batching](https://www.usenix.org/conference/osdi22/presentation/yu) | Why batching is continuous, not static |
| [SARATHI](https://arxiv.org/abs/2308.16369) | Chunked prefill and decode disaggregation |
| [DistServe](https://arxiv.org/abs/2401.09670) / [Splitwise](https://arxiv.org/abs/2311.18677) | Prefill/decode disaggregation |
| [RadixAttention / SGLang](https://arxiv.org/abs/2312.07104) | Prefix caching as a first-class primitive |
| [DeepSpeed-Inference](https://arxiv.org/abs/2207.00032) | Zero-redundancy inference kernels |
| [SpecInfer](https://arxiv.org/abs/2305.09781) / [Medusa](https://arxiv.org/abs/2401.10774) / [EAGLE](https://arxiv.org/abs/2401.15077) | Speculative decoding family |
| [StreamingLLM](https://arxiv.org/abs/2309.17453) / [H2O](https://arxiv.org/abs/2306.14048) / [KIVI](https://arxiv.org/abs/2402.02750) | KV cache management and quantization |

### Efficiency & quantization

| Paper | Why it matters |
|---|---|
| [FlashAttention](https://arxiv.org/abs/2205.14135) · [v2](https://arxiv.org/abs/2307.08691) · [v3](https://arxiv.org/abs/2407.08608) | The canonical memory-hierarchy optimization |
| [LoRA](https://arxiv.org/abs/2106.09685) / [QLoRA](https://arxiv.org/abs/2305.14314) | Parameter-efficient fine-tuning |
| [GPTQ](https://arxiv.org/abs/2210.17323) / [AWQ](https://arxiv.org/abs/2306.00978) / [SmoothQuant](https://arxiv.org/abs/2211.10438) / [LLM.int8()](https://arxiv.org/abs/2208.07339) | Quantization methods you'll actually use |
| [GQA](https://arxiv.org/abs/2305.13245) | Why KV heads ≠ attention heads |

### Scaling, MoE, post-training

| Paper | Why it matters |
|---|---|
| [Switch Transformers](https://arxiv.org/abs/2101.03961) / [GShard](https://arxiv.org/abs/2006.16668) | MoE foundations |
| [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | MoE that works in production |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | MLA, FP8 training, and multi-token prediction in one |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | RL-driven reasoning, openly documented |
| [InstructGPT / RLHF](https://arxiv.org/abs/2203.02155) / [DPO](https://arxiv.org/abs/2305.18290) | Alignment training |
| [Mamba](https://arxiv.org/abs/2312.00752) | The non-attention alternative |

### Retrieval & evaluation

| Paper | Why it matters |
|---|---|
| [RAG (original)](https://arxiv.org/abs/2005.11401) | The baseline pattern |
| [ColBERT](https://arxiv.org/abs/2004.12832) / [SPLADE](https://arxiv.org/abs/2107.05720) | Late interaction and learned sparse retrieval |
| [Lost in the Middle](https://arxiv.org/abs/2307.03172) | Why context ordering matters |
| [Self-RAG](https://arxiv.org/abs/2310.11511) / [CRAG](https://arxiv.org/abs/2401.15884) | Self-correcting retrieval |
| [RAPTOR](https://arxiv.org/abs/2401.18059) / [GraphRAG](https://arxiv.org/abs/2404.16130) | Hierarchical and graph-structured retrieval |

---

## Blogs, newsletters & podcasts

| Source | Focus | Why follow |
|---|---|---|
| [Lil'Log (Lilian Weng)](https://lilianweng.github.io/) | Deep technical surveys | The best long-form ML explainers, period |
| [Ahead of AI (Raschka)](https://magazine.sebastianraschka.com/) | LLM research & practice | Calm, rigorous, reproducible |
| [Interconnects (Nathan Lambert)](https://www.interconnects.ai/) | Post-training & policy | Where RLHF/RLVR is going |
| [SemiAnalysis](https://semianalysis.com/) | Hardware & datacenter economics | The hardware-reality check |
| [Latent Space](https://www.latent.space/) | AI engineering podcast | Practitioner interviews, infra-heavy |
| [Import AI (Jack Clark)](https://importai.substack.com/) | Weekly policy + research | Fast, well-curated |
| [The Batch](https://www.deeplearning.ai/the-batch/) | Weekly AI news | Broad, accessible |
| [Simon Willison's blog](https://simonwillison.net/) | Practical LLM engineering | Honest experimentation, great tooling takes |
| [Eugene Yan](https://eugeneyan.com/) | ML systems in production | Recsys, LLM patterns, hiring |
| [Hamel Husain](https://hamel.dev/) | Evaluation & LLMOps | The best practical eval writing |
| [Horace He](https://horace.io/) | Compilers & performance | The Brrrr intuition |
| [Tim Dettmers](https://timdettmers.com/) | Quantization & hardware | The person behind LLM.int8()/bitsandbytes |
| [Chip Huyen](https://huyenchip.com/blog/) | ML systems design | Framing the whole discipline |
| [GPU MODE](https://github.com/gpu-mode) | Kernels & performance | Community + lectures + Discord |
| [PyTorch blog](https://pytorch.org/blog/) / [vLLM blog](https://blog.vllm.ai/) / [HF blog](https://huggingface.co/blog) | Release engineering | Where the perf wins are announced |
| [Epoch AI](https://epoch.ai/) | Compute trends | Actual numbers on training compute |

---

## Video channels

| Channel | Why |
|---|---|
| [GPU MODE](https://www.youtube.com/@GPUMODE) | Kernel engineering, live |
| [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) | Zero to Hero, nanoGPT, and "Let's build GPT" |
| [Umar Jamil](https://www.youtube.com/@umarjamilai) | Paper walkthroughs with code (attention, LoRA, quantization) |
| [MLSys / MLSys conference](https://mlsys.org/) | Where systems research lands |
| [AI Engineer](https://www.youtube.com/@aiDotEngineer) | Production AI engineering talks |
| [Stanford Online](https://www.youtube.com/@stanfordonline) | CS336/CS224N lectures |
| [Weights & Biases](https://www.youtube.com/@WeightsBiases) | Practical training/serving walkthroughs |
| [Yannic Kilcher](https://www.youtube.com/@YannicKilcher) | Paper critiques |
| [Machine Learning Street Talk](https://www.youtube.com/@MachineLearningStreetTalk) | Long-form debate with researchers |

---

## Benchmarks & leaderboards

| Benchmark | Measures | Use it to |
|---|---|---|
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Broad LM capabilities (MMLU, GSM8K...) | Get reproducible model quality numbers |
| [LMArena](https://lmarena.ai/) | Human preference | Understand perceived quality |
| [MTEB](https://huggingface.co/spaces/mteb/leaderboard) | Embeddings | Pick an embedding model |
| [BEIR](https://github.com/beir-cellar/beir) | Retrieval | Pick a retriever for RAG |
| [SWE-bench](https://github.com/swe-bench/SWE-bench) | Real software tasks | Agentic coding capability |
| [HELM](https://crfm.stanford.edu/helm/) | Holistic multi-metric eval | Avoid single-number thinking |
| [MLPerf](https://mlcommons.org/benchmarks/) | Training & inference throughput | Compare hardware apples-to-apples |
| [InferenceMAX](https://inferencemax.ai/) | Production inference economics | Compare serving stacks and $/token |
| [Chatbot Arena-style human evals](https://lmarena.ai/) | Subjective preference | Sanity-check eval suites |
| [Terminal-Bench](https://github.com/laude-institute/terminal-bench) | Agent terminal tasks | Evaluate agent scaffolds, not just models |

**A warning about leaderboards:** they measure *the benchmark*, not your workload. Contamination, prompt sensitivity, and version drift are real. Always hold out your own eval set — see [P3](PROJECTS.md#p3--fine-tune-with-an-eval-gate) and [P15](PROJECTS.md#p15--eval--red-team-pipeline-in-ci).

---

## Datasets & data sources

| Dataset | Use |
|---|---|
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | High-quality web pretraining data |
| [RedPajama](https://github.com/togethercomputer/RedPajama-Data) | Reproducible open pretraining corpus |
| [The Pile](https://pile.eleuther.ai/) | Diverse text corpus |
| [Dolma](https://huggingface.co/datasets/allenai/dolma) | Open 3T-token corpus with tooling |
| [Common Crawl](https://commoncrawl.org/) | Raw web (you'll need a pipeline) |
| [MS MARCO](https://microsoft.github.io/msmarco/) | Retrieval training/eval |
| [BEIR](https://github.com/beir-cellar/beir) | Retrieval generalization |
| [LibriSpeech](https://www.openslr.org/12) / [Common Voice](https://commonvoice.mozilla.org/) | Speech |
| [LAION](https://laion.ai/) / [COCO](https://cocodataset.org/) | Image-text |
| [GSM8K](https://github.com/openai/grade-school-math) / [MATH](https://github.com/hendrycks/math) | Reasoning eval |

---

## Communities

| Community | Where | Why |
|---|---|---|
| **GPU MODE** | [Discord](https://discord.gg/gpumode) | Kernel/perf help from people who write the kernels |
| **EleutherAI** | [Discord](https://discord.gg/eleutherai) | Open research, evals, training |
| **Hugging Face** | [Discord/forum](https://huggingface.co/join/discord) | Models, datasets, practical help |
| **vLLM** | [Slack/GitHub Discussions](https://github.com/vllm-project/vllm) | Serving issues from the source |
| **SGLang** | [Slack/GitHub](https://github.com/sgl-project/sglang) | Prefix caching, structured output |
| **PyTorch** | [Forum](https://discuss.pytorch.org/) | Framework internals |
| **MLOps Community** | [Slack](https://mlops.community/) | Platform/ops practice |
| **CNCF / Kubernetes** | [Slack](https://slack.cncf.io/) | Scheduling, GPU operator, Kueue |
| **LocalLLaMA** | [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA) | Quantization, consumer hardware, real benchmarks |
| **r/MachineLearning** | [Reddit](https://reddit.com/r/MachineLearning) | Papers and discussion |
| **Latent Space** | [Discord](https://www.latent.space/) | AI engineering practitioners |

---

## Conferences to follow

**Systems:** MLSys · OSDI · SOSP · NSDI · ASPLOS · USENIX ATC · SC · ISCA · Hot Chips
**ML:** NeurIPS · ICML · ICLR · CVPR · ACL · EMNLP
**Data:** SIGMOD · VLDB
**Networking:** SIGCOMM

If you only do one thing: read the **MLSys** proceedings each year. That's where the next generation of infrastructure is published 18 months before it becomes a product.

---

**Back to:** [README](README.md) · [ROADMAP](ROADMAP.md) · [STACK](STACK.md) · [PROJECTS](PROJECTS.md)
