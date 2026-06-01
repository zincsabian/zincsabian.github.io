---
title: AI Performance Engineering 读书笔记
date: 2026-06-01 14:06:00
categories: 读书笔记
tags:
  - 性能优化
  - AI 工程
  - CUDA
---

> 原书地址：https://github.com/cfregly/ai-performance-engineering

## 概述

本书是 O'Reilly 书籍《AI Systems Performance Engineering》的配套代码、工具和资源仓库，涵盖 GPU 优化、分布式训练、推理扩展以及现代 AI 工作负载的全栈性能调优。

现代 AI 系统需要的不仅仅是原始 FLOPs——它们需要以 goodput 为驱动、以分析为先的硬件、软件和算法工程。这本实践指南展示了如何将 GPU、互连网络和运行时堆栈转化为高效、可靠的训练和推理管道。

作者 Chris Fregly 曾在 Netflix、Databricks 和 AWS 推动性能工程创新，并著有《Data Science on AWS》和《Generative AI on AWS》等 O'Reilly 书籍。

## 核心学习要点

- **以 goodput 为目标进行分析** — 使用 Nsight Systems/Compute 和 PyTorch Profiler 找到真正的瓶颈点
- **充分利用内存和带宽** — 优化布局、缓存和数据移动，持续为 GPU 供数
- **编译器调优** — 利用 PyTorch 编译器堆栈和 Triton 生成高影响力内核，无需 C++ 样板代码
- **合理扩展训练** — 应用并行策略（DP、FSDP、TP、PP、CP 和 MoE），重叠计算/通信以最小化气泡
- **高效服务万亿参数模型** — 使用 vLLM、SGLang、TensorRT-LLM 和 NVIDIA Dynamo，结合分离式 Prefill/Decode 和分页 KV Cache
- **降低每 token 成本** — 针对每瓦特性能和每美元吞吐量进行工程优化，而非仅追求峰值速度
- **AI 辅助优化** — 让 AI 帮助合成和调优内核，因为系统规模已超出手动调整的能力

## 章节概览

### 第 1-3 章：基础与系统调优
- AI 系统性能工程师的角色
- 基准测试与分析方法论
- CPU/GPU "Superchip" 架构
- NVIDIA Grace CPU、Blackwell GPU、Tensor Cores
- OS、Docker 和 Kubernetes 调优
- NUMA 感知与 CPU 绑定

### 第 4-5 章：网络与存储
- 重叠通信与计算
- NCCL 分布式多 GPU 通信
- NVIDIA GPUDirect Storage
- 分布式并行文件系统
- 多模态数据处理（NVIDIA DALI）

### 第 6-12 章：GPU 架构与 CUDA 编程
- GPU 架构深入理解（SM、Threads、Warps、Blocks）
- CUDA 编程与内存层次结构
- Roofline 模型分析
- 合并/非合并全局内存访问
- Shared Memory Tiling、Warp Shuffle
- 内核融合、混合精度与 Tensor Cores
- CUTLASS、Inline PTX、SASS 调优
- Intra-Kernel / Inter-Kernel 流水线
- CUDA Graphs、Dynamic Parallelism、NVSHMEM

### 第 13-14 章：PyTorch 与编译器
- PyTorch Profiler 与 NVTX Markers
- torch.compile 深度解析
- OpenAI Triton 自定义内核编写
- PyTorch XLA Backend

### 第 15-19 章：推理优化
- 分离式 Prefill/Decode 架构
- MoE 模型并行策略
- 推测解码与并行解码
- vLLM、SGLang、TensorRT-LLM、NVIDIA Dynamo
- FlashMLA、ThunderMLA、FlexDecoding
- KV Cache 利用率调优
- 动态自适应推理引擎优化

### 第 20 章：AI 辅助性能优化
- AlphaTensor AI 发现的算法
- 自动化 GPU 内核优化
- 自我改进的 AI Agent
- 向百万 GPU 集群扩展

## 200+ 项性能检查清单

本书附带一个涵盖整个生命周期的 200+ 项性能检查清单，包括：
- 性能调优思维与成本优化
- 可复现性与文档最佳实践
- 系统架构与硬件规划
- 操作系统和驱动优化
- GPU 编程与 CUDA 调优
- 分布式训练和网络优化
- 高效推理与服务
- 功耗和热管理
- 最新分析工具和技术
- 架构特定优化

## 社区资源

- 每月举办线上/线下 meetup，覆盖 20+ 城市、100k+ 成员
- YouTube 频道分享技术讲座
- 近期主题包括：动态自适应 RL 推理 CUDA 内核调优、高性能 Agentic AI 推理系统等

## 许可证

Apache 2.0 License
