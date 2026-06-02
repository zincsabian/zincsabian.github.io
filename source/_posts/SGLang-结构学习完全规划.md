---
title: SGLang 结构学习完全规划（基于 RTX 5060 8GB + 云租分布式）
date: 2026-06-02 10:00:00
categories: 读书笔记
tags:
  - SGLang
  - 推理引擎
  - 系统学习
  - 多模态
---

> 本文基于你的实际硬件配置设计：
> - **本地**：RTX 5060 8GB + AMD 9600X
> - **云租**：分布式阶段按需租用多卡
>
> 每步都有 `[ ]` 进度标识，完成后打勾 `[x]`，方便你根据进度提问。

<!-- more -->

---

## 写在前面

### 硬件现实

你的 RTX 5060 有 **8GB 显存**。这意味着：

| 模型大小 | 显存需求 | 能否本地跑 | 推荐度 |
|----------|----------|------------|--------|
| 1B-2B | 3-5 GB | ✅ 轻松 | ⭐⭐⭐⭐⭐ |
| 3B-4B | 6-8 GB | ✅ 能跑 | ⭐⭐⭐⭐⭐ |
| 7B-8B | 14-16 GB | ❌ 不行 | 必须云租 |
| 14B+ | 28GB+ | ❌ 不行 | 必须云租 |

**好消息**：学习 SGLang 的核心架构**不需要大模型**。3B 模型的架构和 70B 完全一样，只是层数/维度不同。理解 Scheduler、KV Cache、Attention 后端的原理与模型大小无关。

**策略**：
- **Phase 0-1**（Day 1-19）：全部本地用 1.5B-3B 模型完成
- **Phase 2**（Day 20-34）：本地读代码理解逻辑，关键实验云租 2-8 卡
- **Phase 3**（Day 35-49）：VLM/扩散小模型本地跑，大模型和 RL 云租

### 进度标识说明

每步开头都有 `[ ]`，完成后改为 `[x]`。你可以：
1. 在博客页面上按 `Ctrl+F` 搜索 `[ ]` 快速找到未完成的步骤
2. 根据当前进度问我问题，例如"我在 Day 5，ZMQ 通信这里卡住了"
3. 我会根据你的进度精准回答，不需要从头解释

---

## Phase 0：环境准备（Day 1-2）

### Day 1：拉代码、建环境、跑通本地单卡

- [ ] 确认已克隆 `~/devroot/sglang`，切到最新 release 分支：`git checkout $(git describe --tags --abbrev=0)`
- [ ] 阅读 `README.md` 的 About 和 Getting Started 两节
- [ ] 按 `docs/get_started/install.md` 装依赖，推荐用 `uv`
- [ ] **📄 精读 [Day 1 源码编译与启动详解](/day1/note)**：理解 `pip install -e`、`sgl-kernel` 编译、`PYTHONPATH` 源码启动、5060 显存控制原理
- [ ] **本地启动 Qwen2.5-1.5B（确保 8GB 不爆显存）**：
  ```bash
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --tp-size 1 \
    --mem-fraction-static 0.85
  ```
- [ ] 用 `curl` 发一条请求验证服务正常
- [ ] 安装 `tmux`、`htop`、`nvtop`，熟悉 GPU 监控
- [ ] 确认 `nvidia-smi` 中显存占用 < 8GB

**做完之后能了解**：
- SGLang 项目整体目录结构（`python/`、`sgl-kernel/`、`sgl-model-gateway/`）
- 如何在你的 RTX 5060 上从源码安装和运行推理服务
- `--mem-fraction-static` 参数对显存控制的作用

**💡 RTX 5060 提示**：8GB 卡启动时建议始终加 `--mem-fraction-static 0.85`，给系统留 1GB 余量，避免 CUDA 初始化时 OOM。

---

### Day 2：目录结构梳理 + CI 初探

- [ ] 阅读 `.github/workflows/` 中任意 2 个 CI yaml，了解测试矩阵
- [ ] 阅读 `test/README.md`，了解测试分类
- [ ] **本地跑通一个集成测试**：
  ```bash
  python -m pytest test/registered/test_xxx.py -v -s \
    --model-path Qwen/Qwen2.5-1.5B-Instruct
  ```
- [ ] 在 `scheduler.py` 加一行日志，重新安装后验证生效
- [ ] 用 `py-spy top --pid <sglang_pid>` 观察运行时线程状态
- [ ] 对比 `nvidia-smi` 中 GPU 利用率 vs `py-spy` 中 Python 线程状态

**做完之后能了解**：
- SGLang 的三层结构边界：Python Runtime、`sgl-kernel`、`sgl-model-gateway`
- 如何在 8GB 卡上修改代码并验证修改生效
- CI 门控规则

---

## Phase 1：单卡核心架构（Day 3-19）

> **全部本地完成**，使用 1.5B-3B 模型。推荐模型矩阵：
> | 实验 | 推荐模型 | 显存占用 | 原因 |
> |------|----------|----------|------|
> | 通用学习 | Qwen2.5-1.5B | ~4GB | 最保险，有余量加日志 |
> | Attention 对比 | Qwen2.5-3B | ~6GB | 足够展示不同后端差异 |
> | 极限测试 | Phi-4-mini (3.8B) | ~7GB | 接近上限，测内存压力 |

---

### Week 1：宏观架构与数据流

#### Day 3：HTTP 入口与请求路由

- [ ] 精读 `python/sglang/srt/entrypoints/http_server.py`
- [ ] 找到 `generate_request`、`chat_completions` 两个路由处理函数
- [ ] 在 `generate_request` 入口处加日志，打印 `request.model_dump_json()`
- [ ] **本地发请求，确认日志输出**：
  ```bash
  curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'
  ```
- [ ] 画出函数调用链：`generate_request` → 哪个函数 → 最终发到哪里

**看完 `http_server.py` 之后能了解**：
- 请求进入 SGLang 的第一个 Python 函数是谁
- FastAPI 路由如何注册（OpenAI API 兼容层）
- 请求体中的字段如何被解析和校验

---

#### Day 4：Engine 类与子进程启动

- [ ] 精读 `python/sglang/srt/entrypoints/engine.py`
- [ ] 找到 `Engine.__init__`，逐行理解：TokenizerManager 创建、Scheduler 子进程 fork、Detokenizer 子进程启动
- [ ] 在 `Engine.__init__` 的每个子进程启动前后加日志
- [ ] **本地启动服务，观察进程树**：`ps aux | grep sglang`，确认有几个进程

**看完 `engine.py` 之后能了解**：
- SGLang 为什么用多进程（Python GIL、CUDA Context 隔离）
- 主进程、Scheduler 进程、Detokenizer 进程的关系

---

#### Day 5：TokenizerManager 与 ZMQ 通信

- [ ] 精读 `python/sglang/srt/managers/tokenizer_manager.py`
- [ ] 找到 `generate_request`，理解它如何把 HTTP 请求转成内部 `Req` 对象
- [ ] 找到 ZMQ socket 创建代码，确认类型（PUSH/PULL/REQ/REP）
- [ ] 在 `generate_request` 和 `recv_from_tokenizer` 两处加日志，观察请求 ID 流转
- [ ] 画一张图：主进程 ↔ Scheduler 进程 ↔ Detokenizer 进程，标注 ZMQ socket 类型

**看完 `tokenizer_manager.py` 之后能了解**：
- Tokenize/Detokenize 在哪个进程执行
- ZMQ 的 PUSH/PULL 模式如何保证请求不丢失

---

#### Day 6：Scheduler 事件循环（上）

- [ ] 精读 `python/sglang/srt/managers/scheduler.py` 的 `event_loop_normal()`，只看前 50 行
- [ ] 理解循环生命周期：init → loop → cleanup
- [ ] 找到 `recv_from_tokenizer()`，理解它如何接收新请求
- [ ] 找到 `schedule()` 的调用点
- [ ] 加日志：每次循环迭代开头打印 `len(self.waiting_queue)`、`len(self.running_queue)`
- [ ] **本地发 3 条请求，观察 queue 长度变化**

**看完 `scheduler.py` 前半之后能了解**：
- Scheduler 是一个什么样的循环
- `waiting_queue` 和 `running_queue` 的区别

---

#### Day 7：Scheduler 事件循环（下）+ 完整追踪

- [ ] 继续精读 `scheduler.py`，理解 `run_batch()` → `process_batch_result()`
- [ ] 找到 `ForwardMode` 枚举，列出所有模式
- [ ] **追踪任务**：在以下位置加日志，发一条请求，收集完整日志链：
  - `http_server.py::generate_request`
  - `tokenizer_manager.py::generate_request`
  - `scheduler.py::event_loop_normal`
  - `scheduler.py::schedule()`
  - `tp_worker.py::forward_batch_generation`
  - `model_runner.py::forward`
  - `detokenizer_manager.py`
- [ ] 按时间戳排序，画出完整调用时序图

**做完追踪之后能了解**：
- 一个 Generate 请求从 HTTP 到 GPU 再回到 HTTP 经历了几次进程切换
- `ForwardMode.PREFILL` 和 `ForwardMode.DECODE` 分别在什么时候触发
- **这是整个 Phase 1 最重要的实验，务必完成**

---

### Week 2：调度策略与内存管理

#### Day 8：ScheduleBatch 数据结构

- [ ] 精读 `python/sglang/srt/managers/schedule_batch.py`
- [ ] 理解 `ScheduleBatch` 字段：`reqs`、`running_bs`、`extend_bs`、`decode_bs`、`input_ids`
- [ ] 理解 `Req` 字段：`origin_input_ids`、`output_ids`、`sample_output_ids`
- [ ] 在 `ScheduleBatch.__init__` 加日志，打印 batch 各字段的 shape
- [ ] **对比 Prefill 和 Decode 时 `input_ids` 的 shape 差异**

**看完 `schedule_batch.py` 之后能了解**：
- `ScheduleBatch` 如何封装一个 step 的所有信息
- `reqs` 和 `running_bs`/`extend_bs`/`decode_bs` 的关系

---

#### Day 9：调度策略对比

- [ ] 精读 `python/sglang/srt/managers/schedule_policy.py`
- [ ] 找到 `LPMSchedulePolicy`、`FCFSSchedulePolicy`、`DFSWeightSchedulePolicy`
- [ ] 理解每种策略的排序逻辑
- [ ] **实验**：修改策略排序函数，调整请求顺序，观察 TTFT/TPOT 变化
- [ ] 思考：为什么默认策略通常是 LPM

**看完 `schedule_policy.py` 之后能了解**：
- 调度策略的本质是"从 waiting_queue 中选择哪些请求进入下一个 step"
- LPM vs FCFS 的 trade-off

---

#### Day 10：Chunked Prefill

- [ ] 在 `scheduler.py` 中搜索 `chunked_prefill_size`
- [ ] 理解长 prefill 如何被切分为多个 chunk
- [ ] 理解 `MIXED` forward mode：同时有 prefill 和 decode
- [ ] **实验**：用一条 4K token 的长 prompt 测试，观察日志中 `ForwardMode` 变化序列
- [ ] 对比启用和不启用 chunked prefill 时长 prompt 的 TTFT

**做完之后能了解**：
- Chunked Prefill 如何让长 prompt 不阻塞短 decode
- `MIXED` mode 的 batch 构成特点

---

#### Day 11：RadixCache 前缀树

- [ ] 精读 `python/sglang/srt/mem_cache/radix_cache.py`
- [ ] 理解 `RadixCache` 的数据结构（RadixTree 节点）
- [ ] 理解 `match_prefix()`、`insert()`、`evict()`
- [ ] **实验**：连续发送两条共享前缀的请求，在 `match_prefix()` 加日志，观察第二次命中 cache 的匹配长度

**看完 `radix_cache.py` 之后能了解**：
- RadixCache 为什么用 radix tree 而不是 hash table
- 缓存命中时哪些 token 可以复用

---

#### Day 12：内存池设计

- [ ] 精读 `python/sglang/srt/mem_cache/memory_pool.py`
- [ ] 理解 `ReqToTokenPool` 和 `TokenToKVPoolAllocator`
- [ ] 理解页分配 vs 连续分配
- [ ] 找到 `MemPoolType` 枚举：MHA、MLA、Hybrid
- [ ] 在 `alloc_token_slots()` 加日志，打印分配的 page 数量
- [ ] 用 `nvidia-smi` 观察显存占用，与内存池统计对比

**看完 `memory_pool.py` 之后能了解**：
- SGLang 的两级内存池设计
- 为什么用页式分配

---

#### Day 13：OOM 与回退机制

- [ ] 在 `scheduler.py` 中搜索 `retract_decode()`
- [ ] 理解动态回退机制：显存不足时将 decode 请求回退到 waiting_queue
- [ ] **实验**：用 Phi-4-mini（接近 8GB 上限）发送大量请求，观察 `retract_decode()` 是否触发
- [ ] 阅读 `docs/advanced_features/hicache_design.md`

**看完 OOM 机制之后能了解**：
- SGLang 如何避免因显存不足导致崩溃
- **在 8GB 卡上，这个机制尤其重要**

---

#### Day 14：本周复盘

- [ ] 不看代码，在白板上画出完整请求流
- [ ] 回答：一个请求从 HTTP 到 GPU 再回到 HTTP 经历了几个进程？
- [ ] 回答：RadixCache 用 radix tree 而非 hash table 的根本原因
- [ ] 回答：`ScheduleBatch` 中 `reqs`、`running_bs`、`extend_bs`、`decode_bs` 的区别
- [ ] 回答：为什么 decode 阶段用 CUDA Graph，而 prefill 不用？
- [ ] 有答不上来的，回到对应文件再读

---

### Week 3：模型执行层与 Attention 后端

#### Day 15：Llama 模型定义

- [ ] 精读 `python/sglang/srt/models/llama.py`
- [ ] 从 `LlamaForCausalLM.forward()` 追踪到 `LlamaAttention.forward()` 再到 `RadixAttention.forward()`
- [ ] 对比 HuggingFace Transformers 的 Llama 实现
- [ ] 在 `LlamaAttention.forward()` 加日志，打印 `hidden_states.shape`

**看完 `llama.py` 之后能了解**：
- SGLang 如何定义一个模型
- `LlamaDecoderLayer` 的结构

---

#### Day 16：RadixAttention 与 Attention 后端注册

- [ ] 精读 `python/sglang/srt/layers/radix_attention.py`
- [ ] 精读 `python/sglang/srt/layers/attention/attention_registry.py`
- [ ] 理解 backend 选择逻辑
- [ ] **实验**（本地，用 Qwen2.5-3B，显存约 6GB）：
  ```bash
  # 测试 FlashInfer
  python -m sglang.launch_server --model-path Qwen/Qwen2.5-3B-Instruct --attention-backend flashinfer
  # 测试 Triton
  python -m sglang.launch_server --model-path Qwen/Qwen2.5-3B-Instruct --attention-backend triton
  ```
- [ ] 用 benchmark 对比两种后端的 prefill/decode 性能

**看完 Attention 注册之后能了解**：
- SGLang 如何支持多种 Attention 后端
- 不同后端的适用条件

---

#### Day 17：线性层与并行策略

- [ ] 精读 `python/sglang/srt/layers/linear.py`
- [ ] 理解 `ColumnParallelLinear` 和 `RowParallelLinear`
- [ ] 理解 `MergedColumnParallelLinear`
- [ ] 对比：ColumnParallel 和 RowParallel 的 all-reduce 时机差异
- [ ] 在 `ColumnParallelLinear.forward()` 加日志，打印 shape

**看完 `linear.py` 之后能了解**：
- 张量并行的线性层实现
- ColumnParallel 先切分后 all-reduce，RowParallel 先计算后 all-reduce

---

#### Day 18：CUDA Graph 与 Decode 加速

- [ ] 精读 `python/sglang/srt/model_executor/cuda_graph_runner.py`
- [ ] 理解 CUDA Graph 捕获和重放机制
- [ ] 理解为什么 batch size 按 2 的幂次捕获
- [ ] **实验**：对比启用和不启用 CUDA Graph 时 decode 的吞吐差异
- [ ] 思考：为什么 prefill 阶段不用 CUDA Graph

**看完 `cuda_graph_runner.py` 之后能了解**：
- CUDA Graph 消除 kernel 启动开销的原理
- 为什么 decode 阶段收益大

---

#### Day 19：Phase 1 综合实验

- [ ] 用 benchmark 跑一轮完整测试：
  ```bash
  python -m sglang.bench_serving \
    --backend sglang \
    --num-prompt 500 \
    --model-path Qwen/Qwen2.5-3B-Instruct
  ```
- [ ] 用 `py-spy` 录制火焰图
- [ ] 对比不同参数：`--attention-backend`、`--chunked-prefill-size`
- [ ] 记录 Phase 1 学习笔记

**做完之后能了解**：
- 如何系统化 benchmark SGLang 性能
- 单卡推理的瓶颈通常在哪里
- **此时你应能独立修改 Scheduler、Attention 后端、线性层，并验证效果**

---

## Phase 2：多卡分布式（Day 20-34）

> **本地读代码理解逻辑，关键实验云租多卡**
> 
> **推荐云租配置**：
> | 实验 | 推荐平台 | 配置 | 预估成本 |
> |------|----------|------|----------|
> | TP=2 入门 | Lambda Cloud | 2×A100 40GB | ~$2.5/小时 |
> | TP=8 整节点 | DataCrunch | 8×A100 80GB | ~$8/小时 |
> | 灵活按需 | RunPod | 2-4×A100 80GB | ~$3-6/小时 |

---

### Week 4：Tensor Parallel

#### Day 20：并行状态初始化（本地读代码）

- [ ] 精读 `python/sglang/srt/distributed/parallel_state.py`
- [ ] 找到 `initialize_model_parallel()`
- [ ] 理解 `--tp-size 8 --dp-size 2 --attn-cp-size 2` 时的 group 划分
- [ ] 画出进程组拓扑图

**看完 `parallel_state.py` 之后能了解**：
- SGLang 如何初始化多个并行进程组
- TP 和 Attention-CP 的区别

---

#### Day 21：通信操作 API（本地读代码）

- [ ] 精读 `python/sglang/srt/distributed/communication_op.py`
- [ ] 列出所有高层通信 API
- [ ] 理解每种操作在 TP/PP/DP 中的使用场景

**看完 `communication_op.py` 之后能了解**：
- 分布式通信的基本原语
- all-reduce 和 all-gather 的区别

---

#### Day 22：双卡 TP 实验（**云租 2×A100**）

- [ ] **租用 2×A100**（Lambda Cloud 或 RunPod，约 $2.5/小时）
- [ ] 启动 TP=2：
  ```bash
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-14B-Instruct \
    --tp-size 2 \
    --host 0.0.0.0 \
    --port 30000
  ```
- [ ] 用 `nvidia-smi` 观察两个 GPU 的显存占用，确认模型被切分
- [ ] 对比 TP=1（本地 3B）和 TP=2（云租 14B）的吞吐差异
- [ ] 在 `parallel_state.py` 加日志，确认 two process groups 已创建

**做完之后能了解**：
- TP=2 时模型权重如何被切分
- 双卡通信开销有多大
- **这是你第一次体验分布式推理**

---

#### Day 23：线性层 All-Reduce 追踪（云租 2×A100）

- [ ] 回到 `linear.py`，追踪 `ColumnParallelLinear` 和 `RowParallelLinear` 的 all-reduce
- [ ] 确认通信时机差异
- [ ] 在 all-reduce 前后加时间戳，测量通信耗时
- [ ] **思考**：为什么这样设计？（数学等价性 + 通信隐藏）

**做完之后能了解**：
- ColumnParallel 和 RowParallel 的通信时机差异
- 通信耗时占 forward 时间的比例

---

#### Day 24：本周复盘

- [ ] 画一张 TP 拓扑图
- [ ] 回答：`parallel_state.py` 中 `_TP`、`_ATTN_TP`、`_MOE_EP`、`_MOE_DP` 的关系
- [ ] 回答：为什么 attention 的 CP 和 TP 要分开

---

### Week 5：Pipeline Parallel

#### Day 25：PP 事件循环（本地读代码）

- [ ] 精读 `python/sglang/srt/managers/scheduler_pp_mixin.py`
- [ ] 找到 `event_loop_pp()`，理解 PP 事件循环
- [ ] 理解 `async_send=True` 和 `_pp_commit_comm_work()`
- [ ] 理解 `forward_stream` 和 `copy_stream` 的双流机制

**看完 `scheduler_pp_mixin.py` 之后能了解**：
- Pipeline Parallel 的事件循环如何调度多个 stage
- 异步通信如何隐藏延迟

---

#### Day 26：PP 层实现（本地读代码）

- [ ] 精读 `python/sglang/srt/layers/pipeline_parallel/`
- [ ] 理解 `PipelineParallelLayer` 的 forward 逻辑
- [ ] 理解 `send_forward()` 和 `recv_forward()`

**做完之后能了解**：
- PP 的层如何在不同 GPU 上分配
- 激活值如何在 stage 之间传输

---

#### Day 27：PP 四卡实验（**云租 4×A100**）

- [ ] **租用 4×A100**（DataCrunch 或 RunPod，约 $5/小时）
- [ ] 启动 PP=4 + TP=1：
  ```bash
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-32B-Instruct \
    --pp-size 4 \
    --tp-size 1
  ```
- [ ] 对比 PP=4 和 TP=4 的吞吐差异
- [ ] 用 `nsys profile` 分析 PP 的 bubble 比例
  ```bash
  nsys profile -o pp_profile \
    python -m sglang.launch_server --model-path ... --pp-size 4
  ```

**做完之后能了解**：
- PP 和 TP 的适用场景差异
- PP 的 bubble 比例对性能的影响
- **这是你第一次用 nsys 分析分布式性能**

---

#### Day 28：本周复盘

- [ ] 画一张 PP=4 的流水线图
- [ ] 回答：PP 的 bubble 在 SGLang 中如何被缓解？
- [ ] 回答：什么时候选 PP、什么时候选 TP？

---

### Week 6：MoE 与 Expert Parallel

#### Day 29：MoE 路由（本地读代码）

- [ ] 精读 `python/sglang/srt/layers/moe/topk.py`
- [ ] 理解 `topk` 路由逻辑
- [ ] 理解 `routing_weights` 和 `selected_experts`

**看完 `topk.py` 之后能了解**：
- MoE 的路由决策如何做出
- top-k 中的 k 是多少

---

#### Day 30：Fused MoE Triton Kernel（本地读代码）

- [ ] 精读 `python/sglang/srt/layers/moe/fused_moe_triton/`
- [ ] 理解 Triton MoE kernel 的输入输出
- [ ] 理解 `dispatch()` → `run_moe_core()` → `combine()`

**看完 Fused MoE 之后能了解**：
- MoE 如何用 Triton 加速
- dispatch 和 combine 阶段的计算逻辑

---

#### Day 31：DeepEP 与 Expert Parallel（本地读代码）

- [ ] 精读 `python/sglang/srt/layers/moe/ep_moe/layer.py`
- [ ] 精读 `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- [ ] 理解 all-to-all 通信
- [ ] 对比：DeepEP 的 all-to-all 和普通 `torch.distributed.all_to_all` 的区别

**看完 DeepEP 之后能了解**：
- Expert Parallel 的通信模式
- DeepEP 如何优化 all-to-all 通信

---

#### Day 32：MoE 双卡实验（**云租 2×A100**）

- [ ] **租用 2×A100**
- [ ] 启动 DeepSeek-V2-Lite（MoE 模型）：
  ```bash
  python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \
    --tp-size 2
  ```
- [ ] 观察 `nvidia-smi`，确认 expert 的负载分布
- [ ] 在 `topk.py` 统计每个 expert 被激活的频率
- [ ] 对比 DeepSeek-V2-Lite 和 Dense 模型的吞吐差异

**做完之后能了解**：
- MoE 模型的实际显存占用
- Expert 负载是否均衡
- **这是你第一次跑 MoE 模型**

---

#### Day 33：Phase 2 综合实验（**云租 8×A100**）

- [ ] **租用 8×A100 整节点**（DataCrunch，约 $8/小时，租 2-3 小时即可）
- [ ] 跑 Qwen2.5-72B：
  ```bash
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-72B-Instruct \
    --tp-size 8
  ```
- [ ] 对比 TP=8 和 TP=4+PP=2 两种配置
- [ ] 用 `nsys profile` 分析通信热点
- [ ] 记录 Phase 2 学习笔记

**做完之后能了解**：
- 70B+ 模型的推理体验
- 整节点 8×A100 的通信模式
- **这是你第一次体验大模型分布式推理**

---

#### Day 34：本周复盘

- [ ] 画一张 MoE 的 all-to-all 通信图
- [ ] 回答：DeepEP 的 all-to-all 和普通 `all_to_all` 有何不同？
- [ ] 回答：如果要支持一种新的 MoE 路由算法，需要修改哪些文件？

---

## Phase 3：多模态与 RL（Day 35-49）

> **多模态小模型本地跑，大模型和 RL 云租**
>
> **本地可跑的 VLM/扩散模型**：
> | 模型 | 大小 | 显存 | 说明 |
> |------|------|------|------|
> | Qwen2.5-VL-3B | 3B | ~6GB | 最佳入门 VLM |
> | LLaVA-1.5-7B（INT4） | 7B | ~5GB | 量化后本地可跑 |
> | Qwen/Qwen-Image | - | ~6GB | 扩散模型 |
>
> **需要云租的**：
> | 模型 | 大小 | 配置 | 说明 |
> |------|------|------|------|
> | Qwen2.5-VL-7B | 7B | 1×A100 | 标准 VLM |
> | Qwen2.5-Math-RM-72B | 72B | 2×A100 | Reward Model |

---

### Week 7：多模态 VLM

#### Day 35：VLM 架构概览（本地读代码）

- [ ] 精读 `python/sglang/srt/models/qwen2_5_vl.py`
- [ ] 理解三段式结构：Vision Encoder → Projector → LLM
- [ ] 对比 `python/sglang/srt/models/llava.py` 的 LLaVA 实现
- [ ] 理解不同 VLM 的视觉编码器差异

**看完 Qwen2.5-VL 之后能了解**：
- VLM 不是"一个模型"，而是"视觉编码器 + 投影层 + 语言模型"的拼接
- 图像 token 如何与文本 token 拼接

---

#### Day 36：Vision Encoder 与图像预处理（本地读代码）

- [ ] 在 `qwen2_5_vl.py` 中找到图像预处理代码
- [ ] 理解图像 resize、patchify、转成视觉 token
- [ ] 打印一张图片的输入 shape

**做完之后能了解**：
- 一张图片被转换成多少个视觉 token
- 为什么不同 VLM 的 patch 数量不同

---

#### Day 37：VLM 本地实验（**RTX 5060，Qwen2.5-VL-3B**）

- [ ] **本地启动 Qwen2.5-VL-3B**：
  ```bash
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-VL-3B-Instruct \
    --tp-size 1 \
    --mem-fraction-static 0.85
  ```
- [ ] 发送图片+文本请求（OpenAI Vision API 格式）
- [ ] 对比纯文本请求和图片请求的延迟差异
- [ ] 在 `model_runner.py` 追踪图片请求的数据流

**做完之后能了解**：
- VLM 的推理延迟主要来自视觉编码器还是 LLM
- **在 8GB 卡上跑 VLM 的体验**

---

#### Day 38：VLM 云租实验（**云租 1×A100，Qwen2.5-VL-7B**）

- [ ] **租用 1×A100**（Lambda Cloud，约 $1.3/小时）
- [ ] 启动 Qwen2.5-VL-7B：
  ```bash
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --tp-size 1
  ```
- [ ] 发送多图请求
- [ ] 对比单图和多图的延迟

**做完之后能了解**：
- 7B VLM 与 3B VLM 的质量差异
- 多图输入的 token 排列方式

---

#### Day 39：本周复盘

- [ ] 画一张 VLM 的数据流图
- [ ] 回答：Qwen2.5-VL 的视觉 token 和文本 token 在 LLM 的输入层如何拼接？
- [ ] 回答：如果要支持视频输入，需要修改哪些文件？

---

### Week 8：扩散模型

#### Day 40：扩散模型管线（本地读代码 + 实验）

- [ ] 阅读 `docs/diffusion/index.md`
- [ ] 安装扩散模块：`uv pip install "sglang[diffusion]" --prerelease=allow`
- [ ] **本地用 RTX 5060 生成一张图片**：
  ```bash
  sglang generate \
    --model-path Qwen/Qwen-Image \
    --prompt "A beautiful sunset" \
    --save-output
  ```
- [ ] 精读 `python/sglang/multimodal_gen/` 目录结构

**看完扩散模块之后能了解**：
- SGLang Diffusion 支持的模型
- 扩散模型的推理管线与 LLM 的差异

---

#### Day 41：TeaCache 加速（本地实验）

- [ ] 阅读 `docs/diffusion/performance/cache/teacache.md`
- [ ] 理解 TeaCache 原理
- [ ] **本地对比启用和不启用 TeaCache 的生成时间**：
  ```bash
  # 启用 TeaCache
  sglang generate --model-path Qwen/Qwen-Image --prompt "..." --use-teacache
  # 不启用
  sglang generate --model-path Qwen/Qwen-Image --prompt "..."
  ```

**做完之后能了解**：
- TeaCache 利用了扩散模型的什么特性来加速
- Cache-DiT 和 TeaCache 的区别

---

#### Day 42：扩散模型深入实验（本地）

- [ ] 启动扩散服务：`sglang serve --model-path Qwen/Qwen-Image --port 30010`
- [ ] 对比不同 prompt 长度、不同分辨率下的生成时间
- [ ] 用 `nvidia-smi` 观察显存占用

**做完之后能了解**：
- 扩散模型的吞吐瓶颈在哪里
- 分辨率对显存和速度的影响

---

#### Day 43：本周复盘

- [ ] 画一张扩散模型的推理流程图
- [ ] 回答：扩散模型和 LLM 在 batching 策略上有什么差异？

---

### Week 9：RL Infra

#### Day 44：Memory Saver（本地读代码）

- [ ] 精读 `python/sglang/srt/utils/torch_memory_saver_adapter.py`
- [ ] 理解 `release_memory_occupation()` 如何释放 KV cache 和 weights
- [ ] 理解 CUDA 虚拟内存机制

**看完 Memory Saver 之后能了解**：
- 为什么 `enable-memory-saver` 能在不杀进程的情况下释放显存
- 这对 RL 的 rollout-training 交替有什么帮助

---

#### Day 45：权重更新（本地读代码）

- [ ] 精读 `python/sglang/srt/managers/scheduler_update_weights_mixin.py`
- [ ] 理解三种更新策略：from_disk、from_tensor、from_distributed
- [ ] 理解 `FlattenedTensorBucket` 的设计

**看完权重更新之后能了解**：
- RL 训练后如何把新权重同步到推理引擎
- 三种更新策略的适用场景

---

#### Day 46-47：最小 RL 循环（**云租 2×A100**）

- [ ] **租用 2×A100**
- [ ] 启动 Policy Model（带 memory saver）：
  ```bash
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-3B-Instruct \
    --tp-size 1 \
    --enable-memory-saver \
    --port 30000
  ```
- [ ] 启动 Reward Model：
  ```bash
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-Math-RM-72B \
    --tp-size 2 \
    --is-embedding-model \
    --port 30001
  ```
- [ ] 写最小 Python 脚本实现完整 RL 循环
- [ ] 观察权重更新后生成结果的变化

**做完之后能了解**：
- RL 训练循环如何与 SGLang 交互
- 权重更新对生成结果的影响
- **这是你第一次搭建完整的 RL Infra**

---

#### Day 48：RL 测试精读

- [ ] 精读 `test/registered/rl/` 下所有 11 个测试
- [ ] 跑通 3 个最重要的测试

**做完之后能了解**：
- SGLang 官方如何测试 RL 功能

---

#### Day 49：全局复盘

- [ ] 回答：为什么 `enable-memory-saver` 能在不杀进程的情况下释放显存？
- [ ] 回答：`update_weights_from_distributed` 中 NCCL group 的生命周期如何管理？
- [ ] 回答：为什么 `deterministic_inference` 对 GRPO 很重要？
- [ ] 全局检查：Phase 0-3 的所有 `[ ]` 是否都变为 `[x]`
- [ ] 写一篇总结博客

---

## 附录：云租平台速查表

| 平台 | 单卡价格 | 多卡价格 | 特点 | 适用场景 |
|------|----------|----------|------|----------|
| **Lambda Cloud** | $1.29/小时 | 2×A100: $2.58/小时 | 最便宜、按秒计费 | TP=2/4 入门 |
| **RunPod** | $1.49/小时 | 4×A100: $6/小时 | 社区镜像丰富 | 多模态实验 |
| **DataCrunch** | - | 8×A100: $8/小时 | 整节点最便宜 | TP=8 大模型 |
| **Vast.ai** | $0.80-1.20/小时 | 按需 | 最便宜 spot | 预算紧张 |
| **AutoDL** | ¥2.50/小时 | 按需 | 国内可用 | 国内用户 |

### 云租启动模板

```bash
# Lambda Cloud / RunPod 通用启动
ssh ubuntu@<your-ip>
sudo apt update && sudo apt install -y git-lfs tmux htop nvtop
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv pip install sglang
```

### 显存速查表

| 模型 | 显存（FP16） | RTX 5060 8GB | 推荐 |
|------|--------------|--------------|------|
| Qwen2.5-1.5B | ~3GB | ✅ 轻松 | Phase 1 首选 |
| Qwen2.5-3B | ~6GB | ✅ 能跑 | Phase 1 进阶 |
| Phi-4-mini (3.8B) | ~7GB | ⚠️ 接近上限 | Phase 1 极限测试 |
| Qwen2.5-VL-3B | ~6GB | ✅ 能跑 | Phase 3 VLM |
| Qwen/Qwen-Image | ~6GB | ✅ 能跑 | Phase 3 扩散 |
| Qwen2.5-7B | ~14GB | ❌ 不行 | 必须云租 |
| Qwen2.5-14B | ~28GB | ❌ 不行 | 云租 TP=2 |
| Qwen2.5-72B | ~144GB | ❌ 不行 | 云租 TP=8 |

---

> **使用指南**：每完成一步就把 `[ ]` 改成 `[x]`，然后告诉我你的进度，例如："我在 Day 11，RadixCache 的 `match_prefix` 返回值我理解不了"。我会基于你的进度精准回答，不需要重复前面内容。
