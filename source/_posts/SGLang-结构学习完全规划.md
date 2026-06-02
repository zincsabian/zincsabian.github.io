---
title: SGLang 结构学习完全规划（含实机部署方案）
date: 2026-06-02 10:00:00
categories: 读书笔记
tags:
  - SGLang
  - 推理引擎
  - 系统学习
  - 多模态
---

> 基于 `~/devroot/sglang` 实际源码和官方文档整理，覆盖纯语言模型与多模态模型，包含详细的实机部署方案和云资源性价比分析。
>
> 目标：从零散理解到能独立修改/扩展 SGLang 任意子系统，具备 RL Infra 开发能力。

<!-- more -->

---

## 一、学习方案总览

SGLang 是一个高吞吐、低延迟的 LLM/VLM 推理引擎，核心架构包含 **Frontend → Runtime → Kernel → Gateway** 四层。以下规划按"单卡理解核心 → 多卡理解分布式 → 多模态/RL 覆盖全栈"的递进路径设计。

| 阶段 | 时长 | 核心目标 | 硬件需求 | 预估成本 |
|------|------|----------|----------|----------|
| **Phase 1** | 2-3 周 | 单卡理解核心架构：Scheduler、KV Cache、Attention | 单卡 24GB+ | ¥300-600 |
| **Phase 2** | 2-3 周 | 多卡理解分布式：TP/PP/DP/EP | 2-8 卡 | ¥800-3000 |
| **Phase 3** | 2-3 周 | 多模态 + RL：VLM、Diffusion、RL Rollout | 2-4 卡 | ¥600-2000 |

> **总预算控制**：¥1700-5600（全部云租方案），如果有本地 RTX 4090，Phase 1 成本可压缩到接近 0。

---

## 二、硬件与云资源方案（性价比分析）

### 2.1 本地方案

| 硬件 | VRAM | 适用阶段 | 成本 | 评价 |
|------|------|----------|------|------|
| RTX 4090 | 24GB | Phase 1 | ¥0（已购） | 性价比之王，8B 模型单卡无压力 |
| RTX 4090 ×2 | 48GB | Phase 1-2 | ¥0 | 可跑 14B/32B 模型 TP=2 |
| A6000 | 48GB | Phase 1-2 | ¥0 | 专业卡，更适合长时间训练/调试 |

**本地方案优势**：零租赁成本、数据安全、随时可用。  
**本地方案劣势**：多卡互联带宽有限（PCIe vs NVLink）、无法体验 8×A100/H100 规模。

### 2.2 云租赁方案对比

| 平台 | 实例类型 | 价格/小时 | 特点 | 推荐度 |
|------|----------|-----------|------|--------|
| **Lambda Cloud** | 1×A100 40GB | ~$1.29 | 价格最低、按秒计费、支持 spot | ⭐⭐⭐⭐⭐ |
| **RunPod** | 1×A100 80GB | ~$1.49 | 社区镜像丰富、网络好、灵活 | ⭐⭐⭐⭐⭐ |
| **Vast.ai** | 1×A100 40GB | ~$0.80-1.20 | 最便宜的 spot 市场、需筛选 | ⭐⭐⭐⭐ |
| **AutoDL** | 1×A100 40GB | ~¥2.50 | 国内可用、按小时计费、镜像全 | ⭐⭐⭐⭐ |
| **DataCrunch** | 8×A100 80GB | ~$7.99 | 整节点便宜、适合 Phase 2 | ⭐⭐⭐⭐ |
| **Nebius** | 8×H100 80GB | ~$15 | H100 最便宜之一、NVLink 全互联 | ⭐⭐⭐⭐ |
| **阿里云** | 1×A100 80GB | ~¥15-25 | 国内合规、企业级 | ⭐⭐⭐ |

**Phase 1 推荐**：Lambda Cloud 1×A100（约 ¥10/小时，每天 4 小时 × 15 天 = ¥600）  
**Phase 2 推荐**：DataCrunch 8×A100 整节点（约 ¥60/小时，按天租用做实验）  
**Phase 3 推荐**：RunPod 2-4×A100（灵活、社区镜像支持多模态）

### 2.3 云租快速启动模板（Lambda Cloud）

```bash
# 1. 注册并创建实例（Ubuntu 22.04 + CUDA 12.4）
# 2. SSH 登录
ssh ubuntu@<your-ip>

# 3. 安装基础依赖
sudo apt update && sudo apt install -y git-lfs tmux htop nvtop

# 4. 安装 uv（比 pip 快 10 倍）
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 5. 安装 SGLang
uv pip install sglang

# 6. 验证安装
python -m sglang.check_env
```

---

## 三、Phase 1：单卡核心架构（第 1-3 周）

### 3.1 目标

- 能在白板上画出请求从 HTTP 到 GPU 再回到用户的完整链路
- 精通 Scheduler 内部状态机
- 理解 RadixCache 前缀树设计
- 能对比 FlashInfer / FlashAttention / Triton 三种 Attention 后端

### 3.2 硬件配置

```
推荐：1×A100 40GB 或 1×RTX 4090 24GB
备用：1×A6000 48GB（本地）
```

### 3.3 模型选择

| 模型 | 大小 | VRAM | 用途 |
|------|------|------|------|
| **Llama-3.1-8B-Instruct** | 8B | ~16GB | 核心学习模型，架构简洁 |
| **Qwen2.5-7B-Instruct** | 7B | ~14GB | 中文支持好，对比学习 |
| **Phi-4-mini** | 3.8B | ~8GB | 极小模型，快速验证 |

**为什么选 Llama-3.1-8B？**
- SGLang 中 `python/sglang/srt/models/llama.py` 是最简洁的参考实现
- 8B 规模刚好能展示所有核心机制，又不会让实验等太久
- 官方 benchmark 大量使用此模型做基准对比

### 3.4 启动指令

```bash
# 基础启动（单卡）
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tp-size 1 \
  --host 0.0.0.0 \
  --port 30000

# 带详细日志启动（用于追踪数据流）
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tp-size 1 \
  --log-level DEBUG \
  --host 0.0.0.0 \
  --port 30000

# 客户端测试
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'
```

### 3.5 核心学习内容（按阅读顺序）

#### Week 1：宏观架构与数据流

| 顺序 | 文件 | 重点 | 验证方式 |
|------|------|------|----------|
| 1 | `python/sglang/srt/entrypoints/http_server.py` | FastAPI 路由注册、请求入口 | 在 `generate_request` 加日志 |
| 2 | `python/sglang/srt/entrypoints/engine.py` | `Engine` 类生命周期、子进程启动 | 断点跟踪 `__init__` |
| 3 | `python/sglang/srt/managers/tokenizer_manager.py` | Tokenize/Detokenize、ZMQ 通信 | 观察 token 长度 |
| 4 | `python/sglang/srt/managers/scheduler.py` | **核心调度器**，事件循环入口 | 这是最重要的文件 |
| 5 | `python/sglang/srt/managers/tp_worker.py` | `TpModelWorker`，TP 组封装 | 理解进程隔离 |
| 6 | `python/sglang/srt/model_executor/model_runner.py` | **实际执行推理**，forward 入口 | 追踪到 CUDA kernel |
| 7 | `python/sglang/srt/managers/detokenizer_manager.py` | 异步 detokenization | 理解异步设计 |

**追踪任务**：一个 Generate 请求的完整链路
1. `http_server.py::generate_request` → 
2. `tokenizer_manager.py::generate_request` → 
3. `scheduler.py::event_loop_normal` → 
4. `tp_worker.py::forward_batch_generation` → 
5. `model_runner.py::forward` → 
6. `detokenizer_manager.py` → HTTP Response

#### Week 2：调度策略与内存管理

| 文件 | 重点 |
|------|------|
| `python/sglang/srt/managers/schedule_policy.py` | LPM、FCFS、DFS_WEIGHT 等策略 |
| `python/sglang/srt/managers/schedule_batch.py` | `ScheduleBatch`、`Req` 数据结构 |
| `python/sglang/srt/managers/scheduler_output_processor_mixin.py` | 输出处理、finish 检测 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 两级内存池 |
| `python/sglang/srt/mem_cache/radix_cache.py` | **RadixCache** 前缀树 |

**实验任务**：
```bash
# 测试 RadixCache 前缀匹配效果
# 发送两个共享前缀的请求，观察第二个是否命中缓存
curl ... -d '{"messages":[{"role":"user","content":"Explain quantum computing"}]}'
curl ... -d '{"messages":[{"role":"user","content":"Explain quantum computing in detail"}]}'
```

#### Week 3：模型执行层与 Attention 后端

| 文件 | 重点 |
|------|------|
| `python/sglang/srt/models/llama.py` | LlamaModel → LlamaDecoderLayer → Attention → MLP |
| `python/sglang/srt/layers/radix_attention.py` | 统一注意力接口 |
| `python/sglang/srt/layers/attention/attention_registry.py` | backend 选择逻辑 |
| `python/sglang/srt/layers/linear.py` | ColumnParallelLinear、RowParallelLinear |

**对比实验**：
```bash
# 对比三种 Attention 后端
python -m sglang.launch_server --model-path ... --attention-backend flashinfer
python -m sglang.launch_server --model-path ... --attention-backend flash_attention
python -m sglang.launch_server --model-path ... --attention-backend triton

# benchmark 对比
python -m sglang.bench_serving --backend sglang --num-prompt 100
```

### 3.6 Phase 1 验证检查点

- [ ] 能在白板上画出完整请求流，包括跨进程的 ZMQ 通信
- [ ] 能解释 `Engine` 和 `Scheduler` 之间为什么需要子进程隔离
- [ ] 能解释一个请求从 `WAITING` → `RUNNING` → `FINISHED` 的状态转换
- [ ] 能画出 RadixCache 的 radix tree 结构
- [ ] 能从 `LlamaForCausalLM.forward()` 追踪到 CUDA kernel 调用
- [ ] 能对比 FlashInfer / FlashAttention / Triton 三种后端的 prefill/decode 性能差异

---

## 四、Phase 2：多卡分布式（第 4-6 周）

### 4.1 目标

- 理解 TP/PP/DP/EP 的实现细节
- 能调试多卡/多机问题
- 能用 nsys 分析通信热点
- 理解 Pipeline Parallel 的 bubble 问题

### 4.2 硬件配置

```
推荐：2-8×A100 80GB（NVLink 互联）
平台：DataCrunch（整节点便宜）或 Lambda Cloud（按小时灵活）
备用：2×RTX 4090（本地，PCIe 互联）
```

### 4.3 模型选择

| 模型 | 大小 | 并行策略 | VRAM | 用途 |
|------|------|----------|------|------|
| **Qwen2.5-72B-Instruct** | 72B | TP=8 | ~144GB | 大模型 TP 实践 |
| **DeepSeek-V2-Lite** | 16B | TP=2 | ~32GB | MoE + MLA 学习 |
| **Qwen3-30B-A3B** | 30B | TP=4 | ~60GB | MoE 路由学习 |
| **Llama-3.1-70B** | 70B | TP=8 | ~140GB | Dense 模型对比 |

**为什么选 DeepSeek-V2-Lite？**
- 是 DeepSeek-V2 的轻量版，保留了 MLA（Multi-Head Latent Attention）和 MoE 架构
- 16B 总参数、2.4B 激活参数，TP=2 即可跑，成本可控
- SGLang 对 DeepSeek 有专门优化，学习价值高

### 4.4 启动指令

```bash
# TP=2（双卡）
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \
  --tp-size 2 \
  --host 0.0.0.0 \
  --port 30000

# TP=8（八卡整节点）
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-72B-Instruct \
  --tp-size 8 \
  --host 0.0.0.0 \
  --port 30000

# PP=4 + TP=2（Pipeline Parallel）
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-72B-Instruct \
  --tp-size 2 \
  --pp-size 4 \
  --host 0.0.0.0 \
  --port 30000

# DP=2 + TP=4（Data Parallel）
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-72B-Instruct \
  --tp-size 4 \
  --dp-size 2 \
  --host 0.0.0.0 \
  --port 30000
```

### 4.5 核心学习内容

#### Week 4：Tensor Parallel 与进程组

| 文件 | 重点 |
|------|------|
| `python/sglang/srt/distributed/parallel_state.py` | `_TP`、`_ATTN_TP`、`_MOE_EP`、`_MOE_DP` 初始化 |
| `python/sglang/srt/distributed/communication_op.py` | all-reduce、all-gather 高层 API |
| `python/sglang/srt/layers/linear.py` | ColumnParallelLinear 和 RowParallelLinear 的 all-reduce 时机差异 |

**关键理解**：
- `ColumnParallelLinear`：all-reduce 在 forward 后
- `RowParallelLinear`：all-reduce 在 forward 前
- 为什么这样设计？（数学等价性 + 通信优化）

#### Week 5：Pipeline Parallel 与 Bubble

| 文件 | 重点 |
|------|------|
| `python/sglang/srt/managers/scheduler_pp_mixin.py` | PP 事件循环、`async_send` |
| `python/sglang/srt/layers/pipeline_parallel/` | PP 层实现 |
| `python/sglang/srt/managers/scheduler.py` | `forward_stream` 和 `copy_stream` 双流机制 |

**实验**：用 nsys 分析 PP 的 bubble
```bash
nsys profile -o pp_profile \
  python -m sglang.launch_server --model-path ... --pp-size 4 --tp-size 2
```

#### Week 6：Expert Parallelism 与 MoE

| 文件 | 重点 |
|------|------|
| `python/sglang/srt/layers/moe/fused_moe_triton/` | Triton MoE kernel |
| `python/sglang/srt/layers/moe/ep_moe/layer.py` | DeepEP MoE |
| `python/sglang/srt/layers/moe/topk.py` | Top-k 路由 |
| `python/sglang/srt/layers/moe/token_dispatcher/` | 各种 EP 后端 |
| `python/sglang/srt/eplb/eplb_manager.py` | Expert Load Balancing |

**实验**：追踪 MoE 路由
```bash
# 启动 DeepSeek-V2-Lite，观察 expert 负载分布
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \
  --tp-size 2 \
  --log-level DEBUG
```

### 4.6 Phase 2 验证检查点

- [ ] 能解释 `--tp-size 8 --dp-size 2 --attn-cp-size 2` 时的 group 划分
- [ ] 能解释 PP 的 bubble 问题，以及 SGLang 如何通过 async P2P 缓解
- [ ] 能用 `nsys profile` 分析一次多卡推理的通信热点
- [ ] 能解释 DeepEP 的 all-to-all 和普通 `torch.distributed.all_to_all` 的区别
- [ ] 能修改一个 layer（如加一个新的 activation）并让 TP 模型正确加载

---

## 五、Phase 3：多模态与 RL（第 7-9 周）

### 5.1 目标

- 理解多模态模型（VLM）的推理管线
- 理解扩散模型的生成管线
- 掌握 SGLang RL 基础设施
- 能独立实现一个 RL 训练循环集成

### 5.2 硬件配置

```
推荐：2-4×A100 80GB
平台：RunPod（多模态社区镜像丰富）
```

### 5.3 模型选择

#### 多模态语言模型（VLM）

| 模型 | 大小 | VRAM | 用途 |
|------|------|------|------|
| **Qwen2.5-VL-7B-Instruct** | 7B | ~16GB | 最佳入门 VLM，架构清晰 |
| **LLaVA-OneVision-7B** | 7B | ~16GB | 对比学习多图输入 |
| **Llama-3.2-11B-Vision** | 11B | ~22GB | 官方 Llama 多模态 |
| **Qwen2.5-VL-72B** | 72B | ~144GB | 大 VLM 分布式实践 |

#### 扩散模型（Diffusion）

| 模型 | 类型 | VRAM | 用途 |
|------|------|------|------|
| **Qwen/Qwen-Image** | 文生图 | ~16GB | 入门扩散 |
| **WAN-AI/Wan2.1-T2V-1.3B** | 文生视频 | ~12GB | 视频生成 |
| **FLUX.1-dev** | 文生图 | ~24GB | 高质量图像 |

#### RL 相关

| 组件 | 用途 |
|------|------|
| **Qwen2.5-Math-RM-72B** | Reward Model 服务 |
| **Llama-3.1-8B-Instruct** | Policy Model |

### 5.4 启动指令

#### VLM 推理

```bash
# Qwen2.5-VL-7B（单卡）
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --tp-size 1 \
  --host 0.0.0.0 \
  --port 30000

# 客户端发送图片+文本请求
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
        {"type": "text", "text": "Describe this image"}
      ]}
    ]
  }'
```

#### 扩散模型推理

```bash
# 安装扩散模块
uv pip install "sglang[diffusion]" --prerelease=allow

# 文生图
sglang generate \
  --model-path Qwen/Qwen-Image \
  --prompt "A beautiful sunset over the mountains" \
  --save-output

# 启动扩散服务
sglang serve --model-path Qwen/Qwen-Image --port 30010
```

#### RL 训练循环

```bash
# 启动 Policy Model（带 memory saver 和 weight update）
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tp-size 1 \
  --enable-memory-saver \
  --host 0.0.0.0 \
  --port 30000

# 启动 Reward Model
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-Math-RM-72B \
  --tp-size 2 \
  --is-embedding-model \
  --host 0.0.0.0 \
  --port 30001
```

### 5.5 核心学习内容

#### Week 7：多模态架构

| 文件 | 重点 |
|------|------|
| `python/sglang/srt/models/qwen2_5_vl.py` | Qwen2.5-VL 实现：Vision Encoder + Projector + LLM |
| `python/sglang/srt/models/llava.py` | LLaVA 实现：CLIP + MLP + LLM |
| `python/sglang/srt/model_executor/model_runner.py` | 多模态输入处理逻辑 |

**关键理解**：
- 图像 → Vision Encoder → 视觉 token → Projector → 与文本 token 拼接 → LLM
- 不同 VLM 的视觉编码器差异：CLIP、SigLIP、ViT

#### Week 8：扩散模型管线

| 文件/目录 | 重点 |
|-----------|------|
| `python/sglang/multimodal_gen/` | 扩散模型核心代码 |
| `python/sglang/multimodal_gen/runtime/post_training/scheduler_rl_mixin.py` | Flow SDE/ODE 的 log-prob 计算 |
| `docs/diffusion/performance/cache/` | Cache-DiT、TeaCache 加速 |

**实验**：对比不同缓存策略
```bash
# 启用 TeaCache
sglang generate --model-path Qwen/Qwen-Image --prompt "..." --use-teacache

# 不启用缓存
sglang generate --model-path Qwen/Qwen-Image --prompt "..."
```

#### Week 9：RL Infra

| 文件 | 重点 |
|------|------|
| `python/sglang/srt/managers/scheduler_update_weights_mixin.py` | 权重更新全链路 |
| `python/sglang/srt/weight_sync/tensor_bucket.py` | Tensor 广播优化 |
| `python/sglang/srt/utils/torch_memory_saver_adapter.py` | 显存睡眠/唤醒 |
| `python/sglang/srt/entrypoints/engine_score_mixin.py` | Reward scoring |
| `test/registered/rl/` | **11 个 RL 测试，必读** |

**实验**：最小 RL 循环
```python
import requests

# 1. 生成 rollout
resp = requests.post("http://localhost:30000/v1/chat/completions", json={
    "model": "default",
    "messages": [{"role": "user", "content": "Solve: 2+3=?"}]
})
rollout = resp.json()["choices"][0]["message"]["content"]

# 2. Score
score_resp = requests.post("http://localhost:30001/v1/score", json={
    "model": "default",
    "text": rollout
})
reward = score_resp.json()["score"]

# 3. Update weights（从训练进程传递新权重）
requests.post("http://localhost:30000/update_weights_from_distributed", ...)
```

### 5.6 Phase 3 验证检查点

- [ ] 能部署 VLM 服务并发送图片+文本请求
- [ ] 能解释 Qwen2.5-VL 的 Vision Encoder → Projector → LLM 数据流
- [ ] 能用 SGLang Diffusion 生成图片并对比 TeaCache 加速效果
- [ ] 能实现最小脚本：启动 SGLang → 生成 rollout → score → update_weights → 继续生成
- [ ] 能解释 `enable-memory-saver` 如何在 CUDA 虚拟内存层面释放显存

---

## 六、与 STUDY_PLAN.md 的映射

本规划与 `~/devroot/sglang/STUDY_PLAN.md` 的 9 Phase 对应关系：

| 本规划 | STUDY_PLAN.md | 内容 |
|--------|---------------|------|
| Phase 1 Week 1 | Phase 1 | 宏观架构与数据流 |
| Phase 1 Week 2 | Phase 2 | 请求生命周期与调度策略 |
| Phase 1 Week 3 | Phase 4 | 模型执行层 |
| Phase 2 Week 4 | Phase 3 (前半) | KV Cache 与内存管理 |
| Phase 2 Week 5 | Phase 5 | 分布式系统：TP/PP |
| Phase 2 Week 6 | Phase 5 (后半) | DP/EP/MoE |
| Phase 3 Week 7 | Phase 8 (前半) | 多模态 |
| Phase 3 Week 8 | Phase 8 (后半) | 扩散模型 |
| Phase 3 Week 9 | Phase 7 | RL Infra |

> **Phase 0**（环境构建）应在本规划开始前完成。  
> **Phase 6**（Kernel 层）和 **Phase 9**（测试/调试）可穿插在各阶段中。

---

## 七、每日学习流程建议

```
1. 晨读（30min）：精读当日目标文件，做注释
2. 代码追踪（60min）：用 IDE 跳转定义，画调用图
3. 实机实验（90min）：修改代码、添加日志、跑测试、看输出
4. 复盘（30min）：写学习笔记，回答当日验证检查点
```

---

## 八、关键问题清单（自我检验）

当你完成全部学习后，尝试不看代码回答这些问题：

1. 一个请求进入 SGLang 后，经历了几个进程？几次序列化/反序列化？
2. RadixCache 用 radix tree 而非 hash table 的根本原因是什么？
3. TP 的 all-reduce 在 `RowParallelLinear` 和 `ColumnParallelLinear` 中分别发生在什么时候？
4. PP 的 bubble 在 SGLang 中如何被缓解？
5. DeepEP 的 all-to-all 和普通的 `torch.distributed.all_to_all` 有何不同？
6. `update_weights_from_distributed` 中 NCCL group 的生命周期如何管理？
7. 为什么 `enable-memory-saver` 能在不杀进程的情况下释放显存？
8. EAGLE 推测解码中，draft model 和 target model 的 KV cache 如何共享/隔离？
9. FlashInfer backend 的 page table 和 vLLM 的 block table 有何异同？
10. 如果要支持一种新的注意力机制（比如 NSA），需要修改哪些文件？
11. Qwen2.5-VL 的视觉 token 和文本 token 在 LLM 的输入层如何拼接？
12. SGLang Diffusion 的 TeaCache 利用了扩散模型的什么特性来加速？
