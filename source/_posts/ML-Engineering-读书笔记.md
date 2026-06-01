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

## 概述

本书是一个开放的方法论、工具和分步指南集合，旨在帮助成功训练和微调大型语言模型（LLM）和多模态模型（VLM），以及相关的推理工作。

这是一本面向 LLM/VLM 训练工程师和操作人员的实用技术资料，包含大量可以直接复制粘贴的脚本和命令，帮助快速解决实际问题。

作者 Stas Bekman 在训练开源 BLOOM-176B 模型（2022年）、IDEFICS-80B 多模态模型（2023年）以及 Contextual.AI 的 RAG 模型（2024年）过程中积累了大量实战经验，并将这些经验整理成这本持续更新的笔记集。

## 主要内容结构

### Part 1. Insights（洞察）

1. **The AI Battlefield Engineering** — 成功所需了解的核心知识
2. **How to Choose a Cloud Provider** — 帮助获得成功云计算体验的关键问题

### Part 2. Hardware（硬件）

1. **Compute** — 加速器、CPU、CPU 内存
2. **Storage** — 本地、分布式和共享文件系统
3. **Network** — 节点内和节点间网络

### Part 3. Orchestration（编排）

1. **Orchestration Systems** — 管理容器和资源
2. **SLURM** — Simple Linux Utility for Resource Management

### Part 4. Training（训练）

1. **Training** — 与模型训练相关的指南

### Part 5. Inference（推理）

1. **Inference** — 模型推理的洞察与实践

### Part 6. Development（开发）

1. **Debugging and Troubleshooting** — 如何调试简单和复杂的问题
2. **And more debugging** — 更多调试方法
3. **Testing** — 大量让测试编写变得愉快的技巧和工具

### Part 7. Miscellaneous（杂项）

1. **Resources** — LLM/VLM 编年史与资源集合

## 实用工具

- **all_reduce_bench.py** — 比 nccl-tests 更简单的网络吞吐量基准测试工具
- **torch-distributed-gpu-test.py** — 快速测试节点间连接性的工具
- **mamf-finder.py** — 测量加速器实际可达 TFLOPS 的工具

## 其他资源

- 提供 PDF 和 EPUB 电子书版本
- 配套有社区 Discussions 供交流
- 作者 Twitter 渠道发布重要更新

## 许可证

Attribution-ShareAlike 4.0 International
