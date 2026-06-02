---
title: ML Engineering 读书笔记（逐日展开版）
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
>
> 这是一本面向 LLM/VLM 训练工程师的开放技术手册。本文按"先读理论 → 再跑实验 → 最后总结"的路径逐日展开，每步都有 `[ ]` 进度标识和预期收获。

<!-- more -->

---

## 写在前面

### 你的硬件能做什么

| 配置 | 能跑什么 | 不能跑什么 |
|------|----------|------------|
| RTX 5060 8GB | 读代码、跑小模型、验证概念 | 训练 7B+ 模型、多卡分布式 |
| CPU AMD 9600X | 数据预处理、编译 | 大规模训练 |

**学习策略**：理解原理不需要大模型。BLOOM-176B 的分布式策略和 1B 模型的原理是一样的，只是规模差异。用本地小模型验证概念，大模型训练的知识通过阅读和思考来掌握。

---

## Part 1：AI 战场工程（Day 1-3）

### Day 1：全书概览与目录梳理

- [ ] 打开 https://github.com/stas00/ml-engineering，阅读 README 的 Table of Contents
- [ ] 下载 PDF/EPUB 版本（电子书链接在 README 中）
- [ ] 快速浏览每个 Part 的标题，标记你最感兴趣的部分
- [ ] 记录：全书共 7 个 Part，你目前对哪个最陌生？

**做完之后能了解**：
- 这本书覆盖的范围：从硬件选型到训练、推理、调试的全流程
- 哪些章节是你知识盲区（比如可能从没接触过 SLURM 或分布式文件系统）
- 全书结构可以作为你未来遇到训练问题时的"查阅地图"

---

### Day 2：AI 竞赛的本质 —— 训练与推理的成本

- [ ] 精读 Part 1 "The AI Battlefield Engineering" 的第一节
- [ ] 列出作者说的 AI 竞赛中最重要的 6 个指标：
  - 训练侧：_____、_____
  - 推理侧：_____、_____、_____
  - 成本侧：_____
- [ ] **思考**：如果你要训练一个 7B 模型，估算一下需要的 GPU 小时数（查一下 Llama-3 的训练成本作为参考）

**做完之后能了解**：
- 为什么训练大模型不是"堆卡就行"，而是**系统工程**
- "买了最快的加速器但在内存/网络/存储上省钱 = 浪费钱" 这句话的底层逻辑

---

### Day 3：ML 工程师的天堂与地狱

- [ ] 精读 Part 1 的 "ML Engineer's heaven and hell"
- [ ] 左边列 "天堂" 的特征，右边列 "地狱" 的特征
- [ ] **对比**：你目前的工作/学习环境更接近天堂还是地狱？
- [ ] 思考：如果你只能改善一个"地狱"条件，优先改善哪个？（通常网络 > 存储 > 计算）

**做完之后能了解**：
- 什么样的基础设施配置能让训练效率最大化
- 为什么"慢速共享文件系统"和"与他人共享网络"是训练瓶颈

---

## Part 2：硬件（Day 4-8）

### Day 4：加速器选型

- [ ] 精读 Part 2 "Compute" 的 Accelerator 章节
- [ ] 整理一张对比表：

| 加速器 | 适用场景 | 优缺点 |
|--------|----------|--------|
| NVIDIA GPU | 通用训练/推理 | 生态最好，最贵 |
| AMD GPU (MI300X) | 大模型训练 | 性价比高，生态建设中 |
| Google TPU | Google Cloud 用户 | 与 TensorFlow/JAX 深度集成 |
| Intel Gaudi | 特定场景 | 价格优势 |

- [ ] 查一下当前 A100/H100/MI300X 的租赁价格（Lambda Cloud、RunPod 等）
- [ ] **本地实验**：用 `nvidia-smi` 查看你的 RTX 5060 的显存带宽和算力
  ```bash
  nvidia-smi -q | grep -E "Product Name|Memory Bandwidth|CUDA Cores|Clock"
  ```

**做完之后能了解**：
- 为什么 GPU 是 ML 训练的首选（并行计算架构适合矩阵乘法）
- 不同加速器的 trade-off：NVIDIA 生态 vs AMD 价格 vs TPU 深度集成

---

### Day 5：存储系统

- [ ] 精读 Part 2 "Storage" 章节
- [ ] 理解 3 种 ML IO 需求：
  1. DataLoader 读取：_____（读/写速度要求）
  2. Checkpoint 写入：_____（读/写速度要求）
  3. 代码加载：_____（读/写速度要求）
- [ ] **本地实验**：测试你的本地存储速度
  ```bash
  # 测试顺序写入速度
  dd if=/dev/zero of=/tmp/test_write bs=1G count=1 oflag=direct
  # 测试顺序读取速度
  dd if=/tmp/test_write of=/dev/null bs=1G count=1 iflag=direct
  # 测试随机读取（模拟 DataLoader）
  fio --name=randread --ioengine=libaio --iodepth=16 --rw=randread --bs=4k --direct=1 --size=1G --numjobs=4
  ```
- [ ] 对比你的存储速度和书中提到的 GPFS/WekaIO/Lustre 的性能差距

**做完之后能了解**：
- 为什么 "用一个超级快的文件系统解决所有问题" 不如 "用 2-3 个不同文件系统分别优化"
- 你的本地 NVMe 速度是否可能成为训练瓶颈（如果是 SSD 而非 NVMe，大概率是）

---

### Day 6：网络架构

- [ ] 精读 Part 2 "Network" 章节
- [ ] 理解 3 种网络：
  - 前端网络（Frontend）：用途 _____，典型速度 _____
  - 后端网络（Backend）：用途 _____，典型速度 _____
  - 带外网络（Out-of-band）：用途 _____
- [ ] 理解关键概念：
  - RDMA：_____
  - InfiniBand：_____
  - NVLink：_____
- [ ] **思考**：如果你只有 2 台服务器，每台 8×A100，用什么互联方案性价比最高？（答案：如果同一机架，NVLink + 100Gbps 以太网；如果跨机架，IB/RoCE）

**做完之后能了解**：
- 为什么网络往往是分布式训练的瓶颈（而非 GPU 算力）
- 单工 vs 双工带宽的区别（书中提到的 600GBps 是双工，实际单向只有 300GBps）

---

### Day 7：编排系统

- [ ] 精读 Part 3 "Orchestration"
- [ ] 对比 SLURM 和 Kubernetes：

| 特性 | SLURM | Kubernetes |
|------|-------|------------|
| 设计目标 | HPC/科学计算 | 容器编排 |
| 学习曲线 | 较陡 | 较平缓 |
| GPU 调度 | 原生支持 | 需要 Device Plugin |
| 适用场景 | 大规模训练 | 推理服务 |

- [ ] **本地实验**：如果你有 SLURM 环境，跑 `sinfo` 和 `squeue` 查看集群状态；如果没有，阅读 `docs/slurm_for_users.md`
- [ ] 理解为什么 "SLURM on Kubernetes"（Slinky/Soperator）是趋势

**做完之后能了解**：
- SLURM 的 job → partition → node → task 层级结构
- 为什么大多数 HPC 集群用 SLURM 而非 K8s

---

### Day 8：硬件周复盘

- [ ] 不看笔记，回答：
  - 如果你要搭建一个 8×A100 的训练集群，存储、网络、编排分别选什么？
  - 为什么 "买最快的 GPU 但配慢速网络" 是浪费？
  - 你的本地机器的瓶颈是什么？（存储？网络？还是计算？）

---

## Part 3：训练（Day 9-12）

### Day 9：模型并行策略

- [ ] 精读 Part 4 "Training" 的 Model Parallelism 章节
- [ ] 画图理解四种并行：
  - **Data Parallel (DP)**：每张卡持有一个完整的模型副本，各自处理不同 batch
  - **Tensor Parallel (TP)**：把一层切分到多张卡，每张卡算一部分
  - **Pipeline Parallel (PP)**：把模型的不同层分配到不同卡
  - **Sequence Parallel (SP)**：把序列维度切分
- [ ] **本地实验**：用 PyTorch 模拟 DP
  ```python
  import torch
  import torch.nn as nn
  from torch.nn.parallel import DataParallel

  model = nn.Linear(10, 10)
  dp_model = DataParallel(model, device_ids=[0])  # 单卡模拟
  input = torch.randn(4, 10).cuda()
  output = dp_model(input)
  print(f"Input shape: {input.shape}, Output shape: {output.shape}")
  ```
- [ ] 理解 FSDP（Fully Sharded Data Parallel）：把模型参数、梯度、优化器状态都分片到各卡

**做完之后能了解**：
- 为什么需要多种并行策略（单一策略无法满足所有需求）
- DP、TP、PP 的适用场景和组合方式

---

### Day 10：性能优化与 Checkpoint

- [ ] 精读 Part 4 的 "Performance" 和 "Checkpoints" 章节
- [ ] 理解混合精度训练（FP16/BF16）的原理
  - FP16：1 位符号 + 5 位指数 + 10 位尾数
  - BF16：1 位符号 + 8 位指数 + 7 位尾数
  - 为什么 BF16 更稳定但精度略低？
- [ ] **本地实验**：对比 FP32、FP16、BF16 的内存占用
  ```python
  import torch
  x_fp32 = torch.randn(1000, 1000)
  x_fp16 = x_fp32.half()
  x_bf16 = x_fp32.bfloat16()
  print(f"FP32: {x_fp32.element_size() * x_fp32.nelement() / 1024**2:.2f} MB")
  print(f"FP16: {x_fp16.element_size() * x_fp16.nelement() / 1024**2:.2f} MB")
  print(f"BF16: {x_bf16.element_size() * x_bf16.nelement() / 1024**2:.2f} MB")
  ```
- [ ] 理解 Checkpoint 策略：
  - 保存频率 vs 训练中断损失
  - 异步 Checkpoint（不阻塞训练）

**做完之后能了解**：
- 混合精度训练如何让速度翻倍、内存减半
- BF16 比 FP16 更适合大模型训练的原因（更大的动态范围）

---

### Day 11：训练不稳定性

- [ ] 精读 Part 4 的 "Instabilities" 章节
- [ ] 列出常见的训练不稳定现象：
  - Loss 爆炸：_____
  - NaN/Inf：_____
  - 梯度消失：_____
- [ ] **本地实验**：用一个小模型故意制造 NaN
  ```python
  import torch
  x = torch.tensor([1e30], requires_grad=True)
  y = x * x  # 1e60，超出 FP32 范围
  print(y)  # 输出 inf
  ```
- [ ] 理解解决方案：
  - 梯度裁剪（Gradient Clipping）
  - 损失缩放（Loss Scaling）
  - 学习率预热（Learning Rate Warmup）

**做完之后能了解**：
- 训练不稳定不是"玄学"，而是可以系统性地诊断和解决的
- 为什么大模型训练特别容易出现 NaN（大矩阵乘法容易数值溢出）

---

### Day 12：训练周复盘

- [ ] 回答：
  - 如果你要训练一个 70B 模型，需要多少 GPU？用什么并行策略组合？
  - 为什么 Checkpoint 要异步保存？
  - 梯度裁剪的阈值怎么选？

---

## Part 4：推理（Day 13-15）

### Day 13：Prefill vs Decode

- [ ] 精读 Part 5 "Inference"
- [ ] 画图理解两个阶段：
  ```
  Prefill:  prompt tokens → 并行处理 → 生成第一个 token
  Decode:   已生成 tokens → 逐个生成 → 直到结束
  ```
- [ ] **本地实验**：用 SGLang 或 Transformers 测量 prefill 和 decode 的延迟差异
  ```bash
  # 用 SGLang 启动一个小模型
  python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct --tp-size 1
  # 发送短 prompt（100 tokens）
  curl ... -d '{"messages":[{"role":"user","content":"Short"}]}'
  # 发送长 prompt（2000 tokens）
  curl ... -d '{"messages":[{"role":"user","content":"Very long prompt..."}]}'
  ```
- [ ] 理解：为什么 prefill 延迟 = f(prompt_length)，decode 延迟 = f(output_length)

**做完之后能了解**：
- 推理延迟的两个独立组成部分
- 为什么 "长 prompt 的首次响应慢" 是正常的（prefill 需要处理整个 prompt）

---

### Day 14：Batching 策略

- [ ] 精读 Part 5 的 "Batching" 章节
- [ ] 对比两种 batching：

| 特性 | Static Batching | Continuous Batching |
|------|-----------------|---------------------|
| 实现复杂度 | 简单 | 复杂 |
| 吞吐 | 低 | 高 |
| 延迟 | 高（等最慢的） | 低（完成即返回） |
| 代表框架 | 早期 vLLM | SGLang、vLLM 0.3+ |

- [ ] **思考**：为什么 Continuous Batching 能显著提高吞吐？（关键：移除已完成请求，立即加入新请求）

**做完之后能了解**：
- Continuous Batching 是现代推理引擎的核心优化
- 为什么它被称为 "in-flight batching"

---

### Day 15：推理周复盘 + 工具链实验

- [ ] 跑通书中提到的三个工具：
  - `all_reduce_bench.py`：测试网络带宽
  - `torch-distributed-gpu-test.py`：测试多卡连通性
  - `mamf-finder.py`：测量实际 TFLOPS
- [ ] **本地实验**：测量 RTX 5060 的实际 FP16 TFLOPS
  ```bash
  # 下载工具
  wget https://raw.githubusercontent.com/stas00/ml-engineering/master/compute/accelerator/mamf-finder.py
  python mamf-finder.py
  ```
- [ ] 对比实测值和理论值（RTX 5060 理论 FP16 算力约 19 TFLOPS）

**做完之后能了解**：
- 你的 GPU 实际能达到多少算力（通常只有理论的 60-80%）
- 为什么实测值低于理论值（内存带宽瓶颈、启动开销等）

---

## Part 5：调试与开发（Day 16-18）

### Day 16：调试方法论

- [ ] 精读 Part 6 "Debugging and Troubleshooting"
- [ ] 掌握两个核心原则：
  1. 快速迭代：_____
  2. 小数据：_____
- [ ] **本地实验**：用小数据复现一个 bug
  ```python
  # 故意制造一个 shape mismatch
  import torch
  a = torch.randn(2, 3)
  b = torch.randn(4, 3)
  c = a + b  # RuntimeError: shape mismatch
  ```
- [ ] 理解：为什么 "用 tiny model 调试" 比 "用 175B model 调试" 高效 100 倍

**做完之后能了解**：
- 调试不是"碰运气"，而是可以系统化的方法论
- "小数据 + 快速迭代" 是调试的万能钥匙

---

### Day 17：PyTorch 调试实战

- [ ] 精读 Part 6 的 "Debugging PyTorch" 章节
- [ ] 掌握三个工具：
  - `torch.autograd.set_detect_anomaly(True)`：自动检测 NaN
  - `torch.cuda.memory_summary()`：显存分析
  - `pdb` 或 `ipdb`：断点调试
- [ ] **本地实验**：用 `detect_anomaly` 定位 NaN 来源
  ```python
  import torch
  torch.autograd.set_detect_anomaly(True)
  x = torch.tensor([1e20], requires_grad=True)
  y = x * x
  z = y.sqrt()  # 可能产生 inf
  z.backward()
  ```
- [ ] 学习 `py-spy` 的使用：`py-spy top --pid <pid>`

**做完之后能了解**：
- PyTorch 的调试工具链
- 如何在训练过程中实时监控显存和性能

---

### Day 18：测试策略

- [ ] 精读 Part 6 的 "Testing" 章节
- [ ] 理解测试分层：
  - 单元测试：测试一个函数
  - 集成测试：测试一组模块
  - 端到端测试：测试完整流程
- [ ] **本地实验**：为一个小函数写测试
  ```python
  import pytest

  def add(a, b):
      return a + b

  def test_add():
      assert add(1, 2) == 3
      assert add(-1, 1) == 0

  if __name__ == "__main__":
      pytest.main([__file__, "-v"])
  ```

**做完之后能了解**：
- 为什么 ML 项目特别需要测试（随机性、数值稳定性）
- 如何用 `pytest` 和 `hypothesis` 写 ML 测试

---

## Part 6：全局复盘（Day 19）

### Day 19：全书总结与自我检验

- [ ] 不看笔记，回答以下问题：
  1. 如果要训练 BLOOM-176B，需要哪些硬件组件？按重要性排序：_____
  2. 为什么 "买 H100 但配 NFS" 是错误决策？_____
  3. RDMA 解决了什么问题？_____
  4. SLURM 的 `sbatch` 和 `srun` 有什么区别？_____
  5. FSDP 和 DDP 的区别是什么？_____
  6. 为什么 Continuous Batching 比 Static Batching 吞吐高？_____
  7. Prefill 和 Decode 哪个对延迟贡献更大？为什么？_____
  8. 调试时为什么先用小数据？_____
- [ ] 记录：这本书对你最有价值的 3 个知识点
- [ ] 记录：你还想深入了解但没覆盖到的 3 个主题

---

## 附录：核心概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| RadixAttention | SGLang 的注意力机制，支持前缀缓存 |
| RDMA | 绕过 CPU，直接读写远程 GPU 内存 |
| InfiniBand | HPC 网络协议，低延迟高带宽 |
| SLURM | HPC 作业调度系统 |
| FSDP | 全分片数据并行，把模型/梯度/优化器状态都分片 |
| Mixed Precision | FP16/BF16 训练，速度翻倍 |
| Gradient Clipping | 限制梯度范数，防止爆炸 |
| Continuous Batching | 动态移除已完成请求，立即加入新请求 |
| Prefill | 一次性处理整个 prompt，生成第一个 token |
| Decode | 逐个生成后续 token |

---

## 附录：本地可复现实验清单

| 实验 | 命令/代码 | 预期结果 |
|------|-----------|----------|
| 测试存储速度 | `fio --name=randread --rw=randread` | 顺序读 > 500MB/s，随机读 > 50MB/s |
| 测量 GPU 算力 | `python mamf-finder.py` | 达到理论值的 60-80% |
| 混合精度内存对比 | `x_fp16 = x_fp32.half()` | FP16 内存减半 |
| 制造 NaN | `x = torch.tensor([1e20]); y = x*x` | 输出 `inf` |
| py-spy 监控 | `py-spy top --pid <pid>` | 看到 Python 线程的 CPU 占用 |
