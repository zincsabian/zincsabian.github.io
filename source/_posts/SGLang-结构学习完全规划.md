---
title: SGLang 结构学习完全规划（逐日展开版）
date: 2026-06-02 10:00:00
categories: 读书笔记
tags:
  - SGLang
  - 推理引擎
  - 系统学习
  - 多模态
---

> 基于 `~/devroot/sglang` 实际源码整理。每步标注"做完/看完 xxx 之后能了解 xxx"，可直接按日执行。

<!-- more -->

---

## 写在前面

本文将学习路径拆到**天级粒度**。每天 2-3 小时，按"读代码 → 做实验 → 写笔记"三段式推进。如果某天完不成，顺延即可，不必赶进度。

> 核心原则：**不理解的地方停下来，加日志、打断点、改代码、看输出**，直到理解为止。不要囫囵吞枣。

---

## Phase 0：环境准备（Day 1-2）

### Day 1：拉代码、建环境、跑通单卡

**动作：**
1. 确认已克隆 `~/devroot/sglang`，切到最新 release 分支：`git checkout $(git describe --tags --abbrev=0)`
2. 阅读 `README.md` 的 About 和 Getting Started 两节
3. 按 `docs/get_started/install.md` 装依赖，推荐用 `uv`
4. 单卡启动 Llama-3.1-8B：`python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --tp-size 1`
5. 用 `curl` 发一条请求验证服务正常
6. 安装 `tmux`、`htop`、`nvtop`，熟悉 GPU 监控

**做完之后能了解：**
- SGLang 项目整体目录结构（`python/`、`sgl-kernel/`、`sgl-model-gateway/`、`benchmark/`、`docs/`）
- 如何从源码安装和运行一个推理服务
- 单卡启动服务的最小命令集

---

### Day 2：目录结构梳理 + CI 初探

**动作：**
1. 阅读 `.github/workflows/` 中任意 2 个 CI yaml，了解测试矩阵（哪些 Python 版本、哪些 GPU、哪些模型）
2. 阅读 `test/README.md`，了解测试分类：`test/srt/`（单元测试）、`test/registered/`（集成测试）、`test/manual/`（手工测试）
3. 跑通一个集成测试：`python -m pytest test/registered/test_xxx.py -v -s`
4. 在 `python/sglang/srt/managers/scheduler.py` 的 `event_loop_normal` 第一行加一行 `logger.info("scheduler started")`，重新安装后验证日志出现
5. 用 `py-spy top --pid <sglang_pid>` 观察运行时线程状态

**做完之后能了解：**
- SGLang 的三层结构边界：Python Runtime、`sgl-kernel`（C++/CUDA）、`sgl-model-gateway`（Rust）
- 如何修改代码并验证修改生效
- CI 的门控规则（什么测试不通过不能合入）

---

## Phase 1：单卡核心架构（Day 3-17）

### Week 1：宏观架构与数据流

#### Day 3：HTTP 入口与请求路由

**动作：**
1. 精读 `python/sglang/srt/entrypoints/http_server.py`
2. 找到 `generate_request`、`chat_completions` 两个路由处理函数
3. 在 `generate_request` 入口处加日志，打印 `request.model_dump_json()`
4. 发一条请求，确认日志输出与预期一致
5. 画出函数调用链：`generate_request` → 哪个函数 → 最终发到哪里

**看完 `http_server.py` 之后能了解：**
- 请求进入 SGLang 的第一个 Python 函数是谁
- FastAPI 路由如何注册（OpenAI API 兼容层）
- 请求体中的字段如何被解析和校验

---

#### Day 4：Engine 类与子进程启动

**动作：**
1. 精读 `python/sglang/srt/entrypoints/engine.py`
2. 找到 `Engine.__init__`，逐行理解：
   - 什么时候创建 `TokenizerManager`
   - 什么时候 `fork`/`spawn` 出 Scheduler 子进程
   - 什么时候启动 Detokenizer 子进程
3. 在 `Engine.__init__` 的每个子进程启动前后加日志
4. 启动服务，观察进程树：`ps aux | grep sglang`，确认有几个进程

**看完 `engine.py` 之后能了解：**
- SGLang 为什么用多进程（Python GIL、CUDA Context 隔离）
- 主进程、Scheduler 进程、Detokenizer 进程的关系
- `TokenizerManager` 在主进程中承担什么角色

---

#### Day 5：TokenizerManager 与 ZMQ 通信

**动作：**
1. 精读 `python/sglang/srt/managers/tokenizer_manager.py`
2. 找到 `generate_request` 方法，理解它如何把 HTTP 请求转成内部 `Req` 对象
3. 找到 ZMQ socket 的创建代码，确认类型（PUSH/PULL/REQ/REP）
4. 在 `generate_request` 和 `recv_from_tokenizer` 两处加日志，观察请求 ID 的流转
5. 画一张图：主进程 ↔ Scheduler 进程 ↔ Detokenizer 进程，标注 ZMQ socket 类型

**看完 `tokenizer_manager.py` 之后能了解：**
- Tokenize/Detokenize 在哪个进程执行
- ZMQ 的 PUSH/PULL 模式如何保证请求不丢失
- `Req` 对象在进程间如何序列化

---

#### Day 6：Scheduler 事件循环（上）

**动作：**
1. 精读 `python/sglang/srt/managers/scheduler.py` 的 `event_loop_normal()`
2. 只看前 50 行：理解循环的生命周期（init → loop → cleanup）
3. 找到 `recv_from_tokenizer()`，理解它如何接收新请求
4. 找到 `schedule()` 的调用点，理解"调度决策"发生在循环的哪个阶段
5. 加日志：在每次循环迭代开头打印 `len(self.waiting_queue)`、`len(self.running_queue)`

**看完 `scheduler.py` 前半之后能了解：**
- Scheduler 是一个什么样的循环（事件驱动还是轮询）
- `waiting_queue` 和 `running_queue` 的区别
- 一个请求进入 Scheduler 后的第一步是什么

---

#### Day 7：Scheduler 事件循环（下）+ 一个请求的完整追踪

**动作：**
1. 继续精读 `scheduler.py`，理解 `run_batch()` → `process_batch_result()` 的流程
2. 找到 `ForwardMode` 枚举，列出所有可能的模式
3. **追踪任务**：在以下位置加日志，发一条请求，收集完整日志链：
   - `http_server.py::generate_request`
   - `tokenizer_manager.py::generate_request`
   - `scheduler.py::event_loop_normal`（循环开始）
   - `scheduler.py::schedule()`（调度决策）
   - `tp_worker.py::forward_batch_generation`
   - `model_runner.py::forward`
   - `detokenizer_manager.py`
   - `tokenizer_manager.py`（HTTP 响应）
4. 按日志时间戳排序，画出完整调用时序图

**做完追踪之后能了解：**
- 一个 Generate 请求从 HTTP 到 GPU 再回到 HTTP 经历了几次进程切换
- 每个阶段耗时多少（哪一步是瓶颈）
- `ForwardMode.PREFILL` 和 `ForwardMode.DECODE` 分别在什么时候触发

---

### Week 2：调度策略与内存管理

#### Day 8：ScheduleBatch 数据结构

**动作：**
1. 精读 `python/sglang/srt/managers/schedule_batch.py`
2. 理解 `ScheduleBatch` 类的字段：`reqs`、`running_bs`、`extend_bs`、`decode_bs`、`input_ids`、`positions`...
3. 理解 `Req` 类的字段：`origin_input_ids`、`output_ids`、`sample_output_ids`...
4. 在 `ScheduleBatch.__init__` 加日志，打印 batch 中各字段的 shape
5. 对比 Prefill 和 Decode 时 `input_ids` 的 shape 差异

**看完 `schedule_batch.py` 之后能了解：**
- SGLang 用 `ScheduleBatch` 封装一个 step 的所有信息
- `reqs` 和 `running_bs`/`extend_bs`/`decode_bs` 的关系
- 为什么 batch 中的请求可能有不同长度的 input

---

#### Day 9：调度策略对比

**动作：**
1. 精读 `python/sglang/srt/managers/schedule_policy.py`
2. 找到 `LPMSchedulePolicy`、`FCFSSchedulePolicy`、`DFSWeightSchedulePolicy`
3. 理解每种策略的排序逻辑
4. **实验**：修改策略排序函数，人为调整请求顺序，观察 TTFT/TPOT 变化
5. 思考：为什么默认策略通常是 LPM（Longest Prefix Match）

**看完 `schedule_policy.py` 之后能了解：**
- 调度策略的本质是"从 waiting_queue 中选择哪些请求进入下一个 step"
- LPM vs FCFS 的 trade-off（prefix cache 命中率 vs 公平性）
- 如何自定义调度策略

---

#### Day 10：Chunked Prefill

**动作：**
1. 在 `scheduler.py` 中搜索 `chunked_prefill_size`
2. 理解长 prefill 如何被切分为多个 chunk
3. 理解 `MIXED` forward mode：一个 batch 中同时有 prefill 和 decode
4. **实验**：用一条 8K token 的长 prompt 测试，观察日志中 `ForwardMode` 的变化序列
5. 对比：启用 chunked prefill 和不启用时，长 prompt 的 TTFT 差异

**做完之后能了解：**
- Chunked Prefill 如何让长 prompt 不阻塞短 decode
- `MIXED` mode 的 batch 构成特点
- 为什么 decode 阶段不需要 chunking

---

#### Day 11：RadixCache 前缀树

**动作：**
1. 精读 `python/sglang/srt/mem_cache/radix_cache.py`
2. 理解 `RadixCache` 的数据结构（RadixTree 节点）
3. 理解 `match_prefix()`：最长前缀匹配的搜索过程
4. 理解 `insert()`：新 token 序列如何插入树
5. 理解 `evict()`：LRU/LFU 淘汰策略
6. **实验**：连续发送两条共享前缀的请求，在 `match_prefix()` 加日志，观察第二次命中 cache 的匹配长度

**看完 `radix_cache.py` 之后能了解：**
- RadixCache 为什么用 radix tree 而不是 hash table（前缀压缩 + 最长匹配）
- 缓存命中时，`match_prefix` 返回什么（哪些 token 可以复用）
- 缓存淘汰策略如何影响命中率

---

#### Day 12：内存池设计

**动作：**
1. 精读 `python/sglang/srt/mem_cache/memory_pool.py`
2. 理解 `ReqToTokenPool` 和 `TokenToKVPoolAllocator`
3. 理解页分配 vs 连续分配
4. 找到 `MemPoolType` 枚举：MHA、MLA、Hybrid
5. 在 `alloc_token_slots()` 加日志，打印分配的 page 数量和位置
6. 用 `nvidia-smi` 观察显存占用，与内存池统计对比

**看完 `memory_pool.py` 之后能了解：**
- SGLang 的两级内存池设计
- 为什么用页式分配而不是逐 token 分配
- 不同注意力机制（MHA/MLA）的内存布局差异

---

#### Day 13：OOM 与回退机制

**动作：**
1. 在 `scheduler.py` 中搜索 `retract_decode()`
2. 理解动态回退机制：当显存不足时，将 decode 请求回退到 waiting_queue
3. 理解 `token_to_kv_pool` 的 `clear()` 逻辑
4. **实验**：发送大量请求使 GPU 接近 OOM，观察 `retract_decode()` 是否触发
5. 阅读 `docs/advanced_features/hicache_design.md`，了解 HiCache 的 CPU/Disk offload

**看完 OOM 机制之后能了解：**
- SGLang 如何避免因显存不足导致崩溃
- `retract_decode` 的触发条件
- HiCache 如何在显存和内存/磁盘之间分层存储 KV

---

#### Day 14：本周复盘 + 白板式验证

**动作：**
1. 不看代码，在白板/纸上画出完整请求流
2. 回答：一个请求从 HTTP 到 GPU 再回到 HTTP 经历了几个进程？几次序列化/反序列化？
3. 回答：RadixCache 的 radix tree 和 trie 有什么区别？为什么选 radix tree？
4. 回答：`ScheduleBatch` 中 `reqs`、`running_bs`、`extend_bs`、`decode_bs` 的区别
5. 回答：为什么 decode 阶段用 CUDA Graph，而 prefill 不用？
6. 如果有答不上来的，回到对应文件再读

**做完之后能了解：**
- 自己对 Week 1-2 内容的掌握程度
- 哪些概念还需要加强

---

### Week 3：模型执行层与 Attention 后端

#### Day 15：Llama 模型定义

**动作：**
1. 精读 `python/sglang/srt/models/llama.py`
2. 从 `LlamaForCausalLM.forward()` 开始追踪：
   - `LlamaModel.forward()` → `LlamaDecoderLayer.forward()`
   - `LlamaAttention.forward()` → `RadixAttention.forward()`
   - `LlamaMLP.forward()` → `MergedColumnParallelLinear` → `RowParallelLinear`
3. 对比 `transformers` 库中的 `LlamaForCausalLM` 实现，找出 SGLang 的差异
4. 在 `LlamaAttention.forward()` 加日志，打印 `hidden_states.shape` 和 `position_ids`

**看完 `llama.py` 之后能了解：**
- SGLang 如何定义一个模型（与 HuggingFace Transformers 的差异）
- `LlamaDecoderLayer` 的结构：Attention → MLP → LayerNorm
- `RadixAttention` 在模型中的位置

---

#### Day 16：RadixAttention 与 Attention 后端注册

**动作：**
1. 精读 `python/sglang/srt/layers/radix_attention.py`
2. 理解 `RadixAttention.forward()` 如何统一调用不同的后端
3. 精读 `python/sglang/srt/layers/attention/attention_registry.py`
4. 理解 backend 选择逻辑（根据什么条件选 FlashInfer / FlashAttention / Triton）
5. 列出 `python/sglang/srt/layers/attention/` 目录下所有 backend 文件
6. **实验**：启动时分别用 `--attention-backend flashinfer`、`flash_attention`、`triton` 启动，观察哪种最快

**看完 Attention 注册之后能了解：**
- SGLang 如何支持多种 Attention 后端
- 不同后端的适用条件（GPU 类型、batch size、sequence length）
- 如何添加一个新的 Attention 后端

---

#### Day 17：线性层与并行策略

**动作：**
1. 精读 `python/sglang/srt/layers/linear.py`
2. 理解 `ColumnParallelLinear` 和 `RowParallelLinear`
3. 理解 `MergedColumnParallelLinear`（用于 MLP 的 gate+up 合并）
4. 对比：ColumnParallel 和 RowParallel 的 all-reduce 时机差异
5. 在 `ColumnParallelLinear.forward()` 加日志，打印输入和输出 shape
6. 理解 `QKVParallelLinear`（用于 Attention 的 qkv 合并）

**看完 `linear.py` 之后能了解：**
- 张量并行的线性层实现
- ColumnParallel 先切分后 all-reduce，RowParallel 先计算后 all-reduce
- 为什么 MLP 的 gate 和 up 要合并成一个矩阵

---

#### Day 18：CUDA Graph 与 Decode 加速

**动作：**
1. 精读 `python/sglang/srt/model_executor/cuda_graph_runner.py`
2. 理解 CUDA Graph 捕获（capture）和重放（replay）机制
3. 理解为什么 batch size 按 2 的幂次捕获（1, 2, 4, 8, 16...）
4. 理解 Breakable CUDA Graph 的实现
5. **实验**：对比启用和不启用 CUDA Graph 时 decode 的吞吐差异
6. 思考：为什么 prefill 阶段不用 CUDA Graph

**看完 `cuda_graph_runner.py` 之后能了解：**
- CUDA Graph 消除 kernel 启动开销的原理
- 为什么 decode 阶段收益大（重复执行、shape 固定）
- SGLang 如何处理 CUDA Graph 捕获失败（动态 shape、同步操作）

---

#### Day 19：Phase 1 综合实验

**动作：**
1. 用 benchmark 跑一轮完整测试：`python -m sglang.bench_serving --backend sglang --num-prompt 1000`
2. 用 `py-spy` 录制火焰图：`py-spy record -o phase1.svg --pid <sglang_pid>`
3. 用 `nsys profile` 分析 GPU 热点
4. 对比不同参数：`--attention-backend`、`--chunked-prefill-size`、`--enable-torch-compile`
5. 记录 Phase 1 学习笔记

**做完之后能了解：**
- 如何系统化地 benchmark SGLang 性能
- 不同参数对吞吐和延迟的影响
- 单卡推理的瓶颈通常在哪里

---

## Phase 2：多卡分布式（Day 20-34）

### Week 4：Tensor Parallel

#### Day 20：并行状态初始化

**动作：**
1. 精读 `python/sglang/srt/distributed/parallel_state.py`
2. 找到 `initialize_model_parallel()`
3. 理解 `--tp-size 8 --dp-size 2 --attn-cp-size 2` 时的 group 划分
4. 画出进程组拓扑图：world_size=16 时，TP/DP/CP 各自包含哪些 rank
5. 理解 `_TP`、`_ATTN_TP`、`_MOE_EP`、`_MOE_DP` 四个全局变量

**看完 `parallel_state.py` 之后能了解：**
- SGLang 如何初始化多个并行进程组
- TP 和 Attention-CP 的区别（为什么需要分开）
- MoE 的 EP（Expert Parallel）和 DP（Data Parallel）如何与 TP 共存

---

#### Day 21：通信操作 API

**动作：**
1. 精读 `python/sglang/srt/distributed/communication_op.py`
2. 列出所有高层通信 API：`all_reduce`、`all_gather`、`reduce_scatter`、`all_to_all`
3. 理解每种操作在 TP/PP/DP 中的使用场景
4. 在 `all_reduce` 加日志，打印 tensor shape 和 group size
5. 对比：NCCL 的 `all_reduce` 和 GLOO 的 `all_reduce` 的差异

**看完 `communication_op.py` 之后能了解：**
- 分布式通信的基本原语
- all-reduce 和 all-gather 的区别
- SGLang 如何封装底层通信库

---

#### Day 22：双卡 TP 实验

**动作：**
1. 云租 2×A100（或本地 2×RTX 4090）
2. 启动 TP=2：`python -m sglang.launch_server --model-path meta-llama/Llama-3.1-70B-Instruct --tp-size 2`
3. 用 `nvidia-smi` 观察两个 GPU 的显存占用，确认模型被切分
4. 对比 TP=1 和 TP=2 的吞吐差异
5. 在 `parallel_state.py` 加日志，确认 two process groups 已创建

**做完之后能了解：**
- TP=2 时模型权重如何被切分（每层分成两份）
- 双卡通信开销有多大
- TP 的吞吐收益和延迟代价

---

#### Day 23：线性层 All-Reduce 追踪

**动作：**
1. 回到 `linear.py`，追踪 `ColumnParallelLinear` 和 `RowParallelLinear` 的 all-reduce 调用
2. 确认：`ColumnParallelLinear` 的 all-reduce 发生在 `forward()` 末尾
3. 确认：`RowParallelLinear` 的 all-reduce 发生在 `forward()` 开头
4. **思考**：为什么这样设计？（数学等价性 + 通信隐藏）
5. 在 all-reduce 前后加时间戳，测量通信耗时

**做完之后能了解：**
- ColumnParallel 和 RowParallel 的通信时机差异
- 为什么 MLP 通常用 Column + Row 的组合
- 通信耗时占 forward 时间的比例

---

#### Day 24：本周复盘

**动作：**
1. 画一张 TP 拓扑图：rank 0 和 rank 1 各自持有哪些层、哪些权重
2. 回答：`parallel_state.py` 中 `_TP`、`_ATTN_TP`、`_MOE_EP`、`_MOE_DP` 的关系
3. 回答：为什么 attention 的 CP（Context Parallel）和 TP 要分开
4. 记录 Phase 2 Week 4 笔记

---

### Week 5：Pipeline Parallel

#### Day 25：PP 事件循环

**动作：**
1. 精读 `python/sglang/srt/managers/scheduler_pp_mixin.py`
2. 找到 `event_loop_pp()`，理解 PP 的事件循环与单卡循环的差异
3. 理解 `async_send=True`：前向传播时异步发送激活值
4. 理解 `_pp_commit_comm_work()`：提交通信工作
5. 理解 `forward_stream` 和 `copy_stream` 的双流机制

**看完 `scheduler_pp_mixin.py` 之后能了解：**
- Pipeline Parallel 的事件循环如何调度多个 stage
- 异步通信如何隐藏延迟
- 双流机制如何 overlap 计算和通信

---

#### Day 26：PP 层实现

**动作：**
1. 精读 `python/sglang/srt/layers/pipeline_parallel/`
2. 理解 `PipelineParallelLayer` 的 forward 逻辑
3. 理解 `send_forward()` 和 `recv_forward()`
4. **实验**：启动 PP=4 + TP=2，用 `nsys profile` 录制时间线
5. 观察：是否存在 bubble（空闲等待）

**做完之后能了解：**
- PP 的层如何在不同 GPU 上分配
- 激活值如何在 stage 之间传输
- Bubble 问题在哪里产生

---

#### Day 27：PP 四卡实验

**动作：**
1. 云租 4×A100
2. 启动 PP=4 + TP=1：`python -m sglang.launch_server --model-path ... --pp-size 4 --tp-size 1`
3. 对比 PP=4 和 TP=4 的吞吐差异
4. 用 `nsys profile` 分析 PP 的 bubble 比例
5. 对比启用 `async_send` 和不启用时的差异

**做完之后能了解：**
- PP 和 TP 的适用场景差异
- PP 的 bubble 比例对性能的影响
- 异步通信的收益

---

#### Day 28：本周复盘

**动作：**
1. 画一张 PP=4 的流水线图，标注每个 stage 的输入输出
2. 回答：PP 的 bubble 在 SGLang 中如何被缓解？
3. 回答：什么时候选 PP、什么时候选 TP？

---

### Week 6：MoE 与 Expert Parallel

#### Day 29：MoE 路由

**动作：**
1. 精读 `python/sglang/srt/layers/moe/topk.py`
2. 理解 `topk` 路由：输入 hidden state → 计算每个 expert 的 score → 选 top-k
3. 理解 `routing_weights` 和 `selected_experts`
4. 在 `topk.py` 加日志，打印每个 token 路由到了哪些 expert

**看完 `topk.py` 之后能了解：**
- MoE 的路由决策如何做出
- top-k 中的 k 是多少（通常是 2 或 6）
- routing weights 如何用于加权求和

---

#### Day 30：Fused MoE Triton Kernel

**动作：**
1. 精读 `python/sglang/srt/layers/moe/fused_moe_triton/`
2. 理解 Triton MoE kernel 的输入输出
3. 理解 `dispatch()` → `run_moe_core()` → `combine()` 的流程
4. 对比 Triton MoE 和 `torch.nn.Linear` 实现的性能差异

**看完 Fused MoE 之后能了解：**
- MoE 如何用 Triton 加速
- dispatch 和 combine 阶段的计算逻辑
- 为什么 MoE 需要专门的 fused kernel

---

#### Day 31：DeepEP 与 Expert Parallel

**动作：**
1. 精读 `python/sglang/srt/layers/moe/ep_moe/layer.py`
2. 理解 `DeepEPMoE.forward()`
3. 精读 `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
4. 理解 all-to-all 通信：每个 GPU 把自己的 token 发送到对应的 expert GPU
5. 对比：DeepEP 的 all-to-all 和普通 `torch.distributed.all_to_all` 的区别

**看完 DeepEP 之后能了解：**
- Expert Parallel 的通信模式
- DeepEP 如何优化 all-to-all 通信
- EP 和 TP 在 MoE 中的分工

---

#### Day 32：MoE 双卡实验

**动作：**
1. 云租 2×A100
2. 启动 DeepSeek-V2-Lite（MoE 模型）：`python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V2-Lite-Chat --tp-size 2`
3. 观察 `nvidia-smi`，确认 expert 的负载分布
4. 对比 DeepSeek-V2-Lite 和 Dense 模型（如 Qwen2.5-7B）的吞吐差异
5. 在 `topk.py` 统计每个 expert 被激活的频率

**做完之后能了解：**
- MoE 模型的实际显存占用（比同参数 Dense 模型低）
- Expert 负载是否均衡
- MoE 的吞吐优势

---

#### Day 33：Phase 2 综合实验

**动作：**
1. 云租 8×A100 整节点
2. 跑 DeepSeek-V2（或 Qwen2.5-72B）在 TP=8 和 TP=4+PP=2 两种配置下
3. 用 benchmark 对比两种配置的吞吐
4. 用 `nsys profile` 分析通信热点
5. 记录 Phase 2 学习笔记

---

#### Day 34：本周复盘

**动作：**
1. 画一张 MoE 的 all-to-all 通信图
2. 回答：DeepEP 的 all-to-all 和普通 `all_to_all` 有何不同？
3. 回答：如果要支持一种新的 MoE 路由算法，需要修改哪些文件？

---

## Phase 3：多模态与 RL（Day 35-49）

### Week 7：多模态 VLM

#### Day 35：VLM 架构概览

**动作：**
1. 精读 `python/sglang/srt/models/qwen2_5_vl.py`
2. 理解 Qwen2.5-VL 的三段式结构：Vision Encoder → Projector → LLM
3. 对比 `python/sglang/srt/models/llava.py` 的 LLaVA 实现
4. 理解不同 VLM 的视觉编码器差异：CLIP、SigLIP、ViT

**看完 Qwen2.5-VL 之后能了解：**
- VLM 不是"一个模型"，而是"视觉编码器 + 投影层 + 语言模型"的拼接
- 图像 token 如何与文本 token 拼接送入 LLM
- 不同 VLM 的架构差异

---

#### Day 36：Vision Encoder 与图像预处理

**动作：**
1. 在 `qwen2_5_vl.py` 中找到图像预处理代码
2. 理解图像如何被 resize、patchify、转成视觉 token
3. 打印一张图片的输入 shape：`(batch, num_patches, hidden_size)`
4. 对比不同分辨率图片的 patch 数量

**做完之后能了解：**
- 一张图片被转换成多少个视觉 token
- 为什么不同 VLM 的 patch 数量不同
- 图像预处理对延迟的影响

---

#### Day 37：VLM 单卡实验

**动作：**
1. 云租 1×A100
2. 启动 Qwen2.5-VL-7B：`python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct --tp-size 1`
3. 发送图片+文本请求（OpenAI Vision API 格式）
4. 对比纯文本请求和图片请求的延迟差异
5. 在 `model_runner.py` 追踪图片请求的数据流

**做完之后能了解：**
- VLM 的推理延迟主要来自视觉编码器还是 LLM
- 图片请求和纯文本请求在 SGLang 中的处理差异

---

#### Day 38：多图输入

**动作：**
1. 精读 `python/sglang/srt/models/llava_onevision.py`
2. 理解 LLaVA-OneVision 的多图输入处理
3. 发送包含多张图片的请求
4. 观察多张图片的视觉 token 如何拼接
5. 对比单图和多图的延迟

**做完之后能了解：**
- 多图输入时 token 如何排列
- 长上下文对多图推理的影响

---

#### Day 39：本周复盘

**动作：**
1. 画一张 VLM 的数据流图：Image → Vision Encoder → Projector → LLM → Text
2. 回答：Qwen2.5-VL 的视觉 token 和文本 token 在 LLM 的输入层如何拼接？
3. 回答：如果要支持视频输入，需要修改哪些文件？

---

### Week 8：扩散模型

#### Day 40：扩散模型管线

**动作：**
1. 阅读 `docs/diffusion/index.md`
2. 安装扩散模块：`uv pip install "sglang[diffusion]" --prerelease=allow`
3. 用 `sglang generate` 生成一张图片
4. 精读 `python/sglang/multimodal_gen/` 目录结构
5. 理解 Diffusion 的 Pipeline：Text Encoder → UNet/DiT → VAE Decoder

**看完扩散模块之后能了解：**
- SGLang Diffusion 支持哪些模型（Wan、Hunyuan、Qwen-Image、FLUX）
- 扩散模型的推理管线与 LLM 的差异
- 为什么扩散模型也需要 batching 和 caching

---

#### Day 41：TeaCache 加速

**动作：**
1. 阅读 `docs/diffusion/performance/cache/teacache.md`
2. 理解 TeaCache 的原理：利用扩散模型的单调收敛特性，跳过部分去噪步
3. 对比启用和不启用 TeaCache 的生成时间
4. 在 `python/sglang/multimodal_gen/runtime/cache/` 找到 TeaCache 实现

**做完之后能了解：**
- TeaCache 利用了扩散模型的什么特性来加速
- Cache-DiT 和 TeaCache 的区别
- 缓存策略对图像质量的影响

---

#### Day 42：扩散模型实验

**动作：**
1. 云租 1×A100
2. 启动扩散服务：`sglang serve --model-path Qwen/Qwen-Image --port 30010`
3. 发送文本生成图片请求
4. 对比不同 prompt 长度、不同分辨率下的生成时间
5. 对比不同模型（Qwen-Image vs FLUX）的性能

**做完之后能了解：**
- 扩散模型的吞吐瓶颈在哪里（UNet/DiT 还是 VAE）
- 分辨率对显存和速度的影响

---

#### Day 43：本周复盘

**动作：**
1. 画一张扩散模型的推理流程图
2. 回答：SGLang Diffusion 的 TeaCache 利用了扩散模型的什么特性？
3. 回答：扩散模型和 LLM 在 batching 策略上有什么差异？

---

### Week 9：RL Infra

#### Day 44：Memory Saver

**动作：**
1. 精读 `python/sglang/srt/utils/torch_memory_saver_adapter.py`
2. 理解 `release_memory_occupation()` 如何释放 KV cache 和 weights
3. 理解 CUDA 虚拟内存机制（保留虚拟地址，释放物理内存）
4. 启动带 `--enable-memory-saver` 的服务
5. 调用 `/release_memory_occupation` 和 `/resume_memory_occupation` API，观察 `nvidia-smi` 显存变化

**看完 Memory Saver 之后能了解：**
- 为什么 `enable-memory-saver` 能在不杀进程的情况下释放显存
- CUDA 虚拟内存和物理内存的区别
- 这对 RL 的 rollout-training 交替有什么帮助

---

#### Day 45：权重更新

**动作：**
1. 精读 `python/sglang/srt/managers/scheduler_update_weights_mixin.py`
2. 理解三种更新策略：from_disk、from_tensor、from_distributed
3. 理解 `FlattenedTensorBucket` 的设计
4. **实验**：用 `update_weights_from_tensor` 更新一个层的权重，观察推理结果变化

**看完权重更新之后能了解：**
- RL 训练后如何把新权重同步到推理引擎
- 三种更新策略的适用场景和性能差异
- `NCCL group` 的生命周期管理

---

#### Day 46：Reward Model 服务

**动作：**
1. 云租 2×A100
2. 启动 Reward Model：`python -m sglang.launch_server --model-path Qwen/Qwen2.5-Math-RM-72B --tp-size 2 --is-embedding-model`
3. 精读 `python/sglang/srt/entrypoints/engine_score_mixin.py`
4. 发送 score 请求，对比 `generate()` 和 `score()` 在 scheduler 中的处理差异

**做完之后能了解：**
- Reward Model 在 RL 流程中的作用
- SGLang 如何同时服务 Policy Model 和 Reward Model
- Score API 与 Generate API 的内部差异

---

#### Day 47：最小 RL 循环

**动作：**
1. 写一个最小 Python 脚本：
   - 向 Policy Model 发送 generate 请求（rollout）
   - 向 Reward Model 发送 score 请求（打分）
   - 向 Policy Model 发送 update_weights_from_tensor（更新权重）
   - 再次发送 generate 请求（下一轮 rollout）
2. 观察权重更新后生成结果的变化
3. 理解 `pause_generation` + `update_weights` + `continue_generation` 的流程

**做完之后能了解：**
- RL 训练循环如何与 SGLang 交互
- 权重更新对生成结果的影响
- 如何实现 mid-rollout 权重更新

---

#### Day 48：RL 测试精读

**动作：**
1. 精读 `test/registered/rl/` 下所有 11 个测试
2. 理解每个测试验证的 RL 功能点
3. 跑通 3 个最重要的测试

**做完之后能了解：**
- SGLang 官方如何测试 RL 功能
- 哪些边界条件需要特别注意

---

#### Day 49：本周复盘 + 全局复盘

**动作：**
1. 回答：为什么 `enable-memory-saver` 能在不杀进程的情况下释放显存？CUDA 虚拟内存机制如何工作？
2. 回答：`update_weights_from_distributed` 中 NCCL group 的生命周期如何管理？
3. 回答：为什么 `deterministic_inference` 对 GRPO 很重要？
4. 全局复盘：回看 Phase 0-3 的所有验证检查点，确认全部通过
5. 写一篇总结博客："我如何花 49 天学会 SGLang"

---

## 附录：可复现指令速查表

### 单卡启动
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tp-size 1 \
  --host 0.0.0.0 \
  --port 30000
```

### TP 启动
```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-72B-Instruct \
  --tp-size 8 \
  --host 0.0.0.0 \
  --port 30000
```

### PP 启动
```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-72B-Instruct \
  --tp-size 2 \
  --pp-size 4 \
  --host 0.0.0.0 \
  --port 30000
```

### VLM 启动
```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --tp-size 1 \
  --host 0.0.0.0 \
  --port 30000
```

### 扩散模型启动
```bash
sglang serve --model-path Qwen/Qwen-Image --port 30010
```

### RL 启动
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tp-size 1 \
  --enable-memory-saver \
  --host 0.0.0.0 \
  --port 30000
```

### Benchmark
```bash
python -m sglang.bench_serving \
  --backend sglang \
  --num-prompt 1000 \
  --request-rate 10
```

### Profile
```bash
nsys profile -o sglang_profile \
  python -m sglang.launch_server --model-path ... --tp-size 8
```

### 客户端测试
```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'
```

---

## 附录：模型选择速查表

| 阶段 | 模型 | 大小 | 并行策略 | VRAM |
|------|------|------|----------|------|
| Phase 1 | Llama-3.1-8B-Instruct | 8B | TP=1 | 16GB |
| Phase 1 | Qwen2.5-7B-Instruct | 7B | TP=1 | 14GB |
| Phase 2 | DeepSeek-V2-Lite | 16B | TP=2 | 32GB |
| Phase 2 | Qwen2.5-72B-Instruct | 72B | TP=8 | 144GB |
| Phase 3 | Qwen2.5-VL-7B-Instruct | 7B | TP=1 | 16GB |
| Phase 3 | Qwen/Qwen-Image | - | TP=1 | 16GB |
| Phase 3 | Qwen2.5-Math-RM-72B | 72B | TP=2 | 144GB |
