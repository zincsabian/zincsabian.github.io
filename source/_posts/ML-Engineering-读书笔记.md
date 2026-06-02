---
title: ML Engineering 读书笔记
date: 2026-06-01 14:05:00
categories: 读书笔记
tags:
  - 机器学习
  - 工程实践
  - LLM
---

> 原书地址：https://github.com/stas00/ml-engineering
>
> 作者：Stas Bekman（BLOOM-176B、IDEFICS-80B 训练工程师）
> 许可：CC BY-SA 4.0

这是一本面向 LLM/VLM 训练工程师和操作人员的开放技术手册，涵盖从硬件选型到训练、推理、调试的全流程实战经验。以下内容按照原书结构整理，方便日后查阅和学习。

<!-- more -->

---

## 一、AI 战场工程（Insights）

### 1.1 AI 竞赛中什么最重要？

**训练阶段：**
1. 多快能训练出一个更好的模型（先发优势）
2. 花了多少钱（训练后是否还有钱付工资？）

**推理阶段：**
1. 低延迟（用户习惯毫秒级响应，秒级会流失）
2. 高吞吐（能同时处理多少查询）
3. 每个用户的成本（能否租更多 GPU 来服务更多用户？）

### 1.2 LLM 训练需要什么？

1. **快速的计算能力** —— 主要由矩阵乘法主导
2. **足够快的内存、IO、网络和 CPU** —— 来为计算单元供数

**推论：** 如果你买了最快的加速器，但在其他组件上省钱，那就是浪费钱，训练会更慢。

### 1.3 ML 的"主力军"

- **加速器**：GPU、TPU、IPU、FPGA、HPU、QPU、RDU 等
- 最近的 CPU 也越来越多地被用于推理

### 1.4 信息分享文化

AI 领域有一个令人惊讶的现象：几乎所有人都会与社区分享大量发现。虽然公司不会披露所有 IP，但很多知识或模型权重都会公开。
- Twitter 是跟踪最新动态的核心平台

### 1.5 ML 工程师的天堂与地狱

**天堂：**
- 有人专门维护硬件和系统的 HPC/全托管云集群
- 大量可独占使用的节点
- 快速的节点间连接，不与他人共享
- 巨大的本地 NVMe 共享文件系统
-  barebones Linux + SLURM，最小化软件栈
- 拥有 `sudo` 权限

**地狱：**
- 需要自己当系统管理员的云/内部集群
- 缓慢的小型共享文件系统（NFS？）
- 缓慢的节点间网络导致加速器利用率低
- 网络与其他用户共享，导致不稳定
- 超级复杂的云控制台，连简单设置都要点无数个屏幕

---

## 二、硬件（Hardware）

### 2.1 计算（Compute）

**加速器（Accelerator）**：ML 的主力军
- GPU、TPU、IPU、FPGA、HPUs、QPUs、RDUs
- 选择加速器时需关注：理论 TFLOPS、内存大小和带宽

**CPU**：
- CPU 亲和性（affinity）设置很重要
- 需要足够快的 CPU 来预处理数据并调度加速器

**CPU 内存**：
- "多少 CPU 内存才够？" —— 这是全书最短的章节
- 答案：越多越好，但关键是平衡

### 2.2 存储（Storage）

ML 工作负载有 3 种截然不同的 IO 需求：

1. **DataLoader 读取** —— 需要超快的可持续读取，持续数小时甚至数天
2. **Checkpoint 写入** —— 需要超快的突发写入，越快越好，否则会阻塞训练
3. **代码加载** —— 中等速度读写，需要共享以便所有节点看到相同代码

**理想的文件系统选择：**

分布式并行文件系统是最高效的解决方案：
- **GPFS**（IBM，现称 IBM Storage Scale）
- **WekaIO**
- **Lustre FS**（开源）

其他选择：
- BeeGFS
- DAOS（Intel）
- NetApp、VAST

**案例研究：** 在 JeanZay HPC（法国），2021 年用 GPFS + NVMe 在 384 个进程上并行保存 2.3TB checkpoint 只用了 40 秒。

### 2.3 网络（Network）

节点有 3 种网络，速度差异很大：

1. **前端网络（Frontend）** —— 互联网连接、分布式存储、编排（SLURM/K8s），通常 100-400Gbps
2. **后端网络（Backend）** —— 加速器间通信，需要极高的带宽
3. **带外网络（Out-of-band）** —— 管理/监控用途

**关键概念：**
- **RDMA**（Remote Direct Memory Access）：绕过 CPU 直接读写远程内存
- **RoCE**（RDMA over Converged Ethernet）
- **InfiniBand（IB）**：高性能计算网络的事实标准
- **NVLink/NVSwitch**：NVIDIA GPU 之间的高速互联

**单工 vs 双工：** 注意区分单向（Unidirectional）和双向（Duplex）带宽，后者通常是前者的 2 倍。

---

## 三、编排（Orchestration）

### 3.1 SLURM

Simple Linux Utility for Resource Management，在大多数 HPC 环境中都能找到，已有 20 多年历史。

**SLURM on Kubernetes：**
- **Slinky**：SLURM 原作者开发的框架
- **Soperator**：Nebius 出品

### 3.2 Kubernetes

K8s，最流行的容器编排系统，用于自动化部署、扩展和管理容器化应用。

### 3.3 其他编排方案

- **dstack**：K8s 和 Slurm 的轻量级开源替代，支持 NVIDIA、AMD、TPU
- **SkyPilot**：统一执行、高成本节省、高 GPU 可用性
- **OpenHPC**：提供 HPC Linux 集群所需的各种预构建组件
- **run.ai**：被 NVIDIA 收购，计划开源
- **Docker Swarm**：Docker 官方的容器编排工具

---

## 四、训练（Training）

### 4.1 模型并行（Model Parallelism）

当模型太大，单个加速器放不下时的策略：
- **数据并行（DP）**：每个 GPU 持有完整模型副本，处理不同数据批次
- **张量并行（TP）**：将模型层切分到多个 GPU
- **流水线并行（PP）**：将模型按层分配到不同 GPU
- **FSDP**：Fully Sharded Data Parallel，PyTorch 的分布式方案

### 4.2 性能优化

- 重叠计算与通信
- 使用混合精度（FP16/BF16）加速训练
- 优化数据加载器，避免 GPU 等待数据

### 4.3 容错（Fault Tolerance）

- 频繁保存 checkpoint
- 使用弹性训练框架（如 Elastic Horovod、Torch Elastic）
- 监控节点健康状态

### 4.4 不稳定性（Instabilities）

- Loss 爆炸：学习率过高、梯度裁剪不当
- NaN/Inf：数值下溢/上溢
- 使用 `--detect-anomaly` 或自定义检测工具

### 4.5 Checkpoint

- 保存频率：平衡恢复成本与丢失进度
- 异步保存：避免阻塞训练
- 案例：2.3TB checkpoint 在 384 进程上 40 秒完成（GPFS + NVMe）

---

## 五、推理（Inference）

### 5.1 核心概念

**Prefill 与 Decode：**
- **Prefill**：一次性处理完整 prompt（类似训练），构建 KV Cache。延迟很小。
- **Decode**：逐个生成新 token（回归式），无法并行化，是延迟的主要来源。

**Online vs Offline 推理：**
- **Online**：实时用户查询（聊天机器人、搜索引擎），需要低 TTFT（首 token 时间）和低延迟
- **Offline**：批量处理（基准测试、合成数据生成），需要高吞吐

**Grounding（ grounding/上下文）：**
- 给预训练模型提供训练时未见过的额外信息
- 主要技术：**RAG**（检索增强生成）、微调

### 5.2 Batching

**静态批处理（Static Batching）：**
- 将前 N 个查询 batch 在一起
- 问题：如果某个查询提前完成，必须等最慢的完成才能返回

**动态批处理（Continuous/In-flight Batching）：**
- 完成的查询立即移除，新查询立即加入
- batch 中不同位置可能处于完全不同的生成阶段
- 显著提高响应速度，是 vLLM 等推理引擎的核心优化

---

## 六、开发与调试（Development）

### 6.1 快速调试方法论

**两个核心需求：**
1. **快速迭代**：从重启到关键点的等待时间应控制在几秒
2. **小数据**：用最小数据调试，更容易记住、比较和心算

**数据选择策略：**
- 程序崩溃时：任何随机/合成数据都可以
- 程序能跑但质量不对：需要真实数据
- 用 `rand` 或精心设计的合成数据（如 `[[1.0, 2.0], [3.0, 4.0]]`）
- ML 调试：用 10K 参数模型调功能，1B 模型测质量，175B 模型做最终训练

### 6.2 调试 PyTorch 程序

- 使用 `torch-distributed-gpu-test.py` 检查所有 GPU 是否能互相通信
- 诊断 Hanging 和 Deadlock（多节点多 GPU Python 程序）
- 检测 Underflow/Overflow
- 使用 `NicerTrace.py` 改进 trace 输出

### 6.3 网络调试

- 使用 `all_reduce_bench.py` 替代 nccl-tests 进行网络吞吐量基准测试
- 检查 RDMA 连接状态
- 验证 NCCL 环境变量设置

### 6.4 测试

- 编写可复现的测试
- 使用 tiny models/tokenizers/datasets 加速测试
- 持续集成（CI）确保代码质量

---

## 七、实用工具

| 工具 | 用途 |
|------|------|
| `all_reduce_bench.py` | 网络吞吐量基准测试（比 nccl-tests 更简单） |
| `torch-distributed-gpu-test.py` | 快速测试节点间 GPU 连通性 |
| `mamf-finder.py` | 测量加速器实际可达 TFLOPS |
| `printflock.py` | 多 GPU 环境下非交错打印 |
| `NicerTrace.py` | 改进的 Python trace 模块 |

---

## 八、资源与更新

- **电子书**：提供 PDF 和 EPUB 格式
- **社区讨论**：GitHub Discussions 供交流
- **更新通知**：作者 Twitter @StasBekman
- **许可证**：Attribution-ShareAlike 4.0 International
