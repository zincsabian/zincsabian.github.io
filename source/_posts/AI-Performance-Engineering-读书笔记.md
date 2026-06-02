---
title: AI Performance Engineering 读书笔记（逐日展开版）
date: 2026-06-01 14:06:00
categories: 读书笔记
tags:
  - 性能优化
  - AI 工程
  - CUDA
---

> 原书地址：https://github.com/cfregly/ai-performance-engineering
>
> 作者：Chris Fregly（Netflix、Databricks、AWS 背景，O'Reilly 作者）
> 许可：Apache 2.0
>
> 本书是 O'Reilly 书籍配套资源，覆盖从 GPU 微架构到分布式推理的全栈性能优化。本文按"先理解原理 → 再跑实验 → 最后分析"的路径逐日展开。

<!-- more -->

---

## 写在前面

### 本书与你的 RTX 5060

这本书大量涉及 CUDA 编程和 GPU 微架构。你的 RTX 5060 虽然只有 8GB 显存，但**完全足够学习和实验**：

| 实验类型 | RTX 5060 能否完成 | 说明 |
|----------|-------------------|------|
| CUDA 编程练习 | ✅ | 8GB 足够跑所有教程 kernel |
| Nsight 分析 | ✅ | 单卡分析不受显存限制 |
| PyTorch Profiler | ✅ | 任意模型都可分析 |
| Triton Kernel | ✅ | 8GB 足够编译和运行 |
| 分布式训练 | ❌ | 需要多卡，后续云租 |
| 70B 模型推理 | ❌ | 需要 140GB+，后续云租 |

**学习策略**：CUDA 编程和性能分析的原理在所有 NVIDIA GPU 上都是一样的。在 RTX 5060 上学到的 nsys 分析技巧，拿到 H100 上完全一样用。

---

## 第一部分：基础与系统（Day 1-5）

### Day 1：性能工程思维

- [ ] 阅读本书 README 的 "核心学习要点" 部分
- [ ] 列出 8 个核心学习目标中的前 4 个：
  1. goodput-driven 分析：_____
  2. 内存带宽优化：_____
  3. 编译器调优：_____
  4. 分布式训练扩展：_____
- [ ] **思考**："Profile for goodput, not just utilization" 是什么意思？（提示：GPU 利用率 100% 不代表算力被有效利用，可能是内存等待）
- [ ] **本地实验**：用 `nvidia-smi dmon` 监控你的 RTX 5060，观察利用率、显存带宽、温度
  ```bash
  # 每 1 秒采样一次，显示 SM 利用率、显存带宽、功耗
  nvidia-smi dmon -s umt
  ```

**做完之后能了解**：
- 性能工程不是"把 GPU 利用率跑满"，而是"让有效计算最大化"
- 为什么 memory-bound 的程序 GPU 利用率也会很高（GPU 在等待内存，不是在做计算）

---

### Day 2：GPU 架构入门

- [ ] 精读 Chapter 2 "AI System Hardware Overview"
- [ ] 理解 GPU 核心组件：
  - **Streaming Multiprocessor (SM)**：GPU 的基本计算单元，包含多个 CUDA Core
  - **Tensor Core**：专门做矩阵乘法的硬件单元（比 CUDA Core 快 8-16 倍）
  - **Warp**：32 个线程为一组，同一 warp 内的线程执行相同指令
  - **Shared Memory**：SM 内的快速缓存，比全局内存快 ~100 倍
  - **L2 Cache**：所有 SM 共享
  - **HBM（全局内存）**：最大但最慢
- [ ] **本地实验**：查看你的 GPU 的 SM 数量和显存带宽
  ```bash
  nvidia-smi -q | grep -E "Product Name|SMs|Memory Bandwidth|L2 Cache"
  # 或者用 deviceQuery（CUDA Samples）
  /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery
  ```
- [ ] 理解 Roofline 模型的概念：判断程序是 compute-bound 还是 memory-bound

**做完之后能了解**：
- GPU 不是"很多 CPU 核"，而是 SIMD 架构（单指令多数据）
- 为什么 Tensor Core 对 Transformer 训练如此重要（注意力机制就是大量矩阵乘法）

---

### Day 3：Roofline 模型分析

- [ ] 理解 Roofline 模型：
  ```
  性能上限 = min(峰值算力, 峰值带宽 × 算术强度)
  ```
  - 算术强度 = 计算量 / 内存访问量
  - Compute-bound：程序在 Roofline 的平顶区域
  - Memory-bound：程序在 Roofline 的斜线区域
- [ ] **本地实验**：用简单的 PyTorch 操作对比 compute-bound 和 memory-bound
  ```python
  import torch
  import time

  device = 'cuda'
  n = 4096

  # Compute-bound: 大矩阵乘法
  a = torch.randn(n, n, device=device)
  b = torch.randn(n, n, device=device)
  torch.cuda.synchronize()
  t0 = time.time()
  c = torch.matmul(a, b)
  torch.cuda.synchronize()
  print(f"Matmul: {(time.time()-t0)*1000:.2f} ms")

  # Memory-bound: 逐元素加法（算数强度低）
  x = torch.randn(n*n, device=device)
  torch.cuda.synchronize()
  t0 = time.time()
  y = x + 1.0
  torch.cuda.synchronize()
  print(f"Add: {(time.time()-t0)*1000:.2f} ms")
  ```
- [ ] 计算两种操作的算术强度，判断各自在 Roofline 的哪个区域

**做完之后能了解**：
- 如何判断你的 kernel 是 compute-bound 还是 memory-bound
- 优化方向完全不同：compute-bound 要优化计算，memory-bound 要优化数据复用

---

### Day 4：Nsight 工具链

- [ ] 阅读 Chapter 3 "OS, Docker, and Kubernetes Tuning" 中关于 profiling 的部分
- [ ] 安装 Nsight Systems：`pip install nvidia-nsys`（通常随 CUDA 安装）
- [ ] **本地实验**：用 Nsight Systems 分析一个 PyTorch 程序
  ```bash
  # 录制时间线
  nsys profile -o my_profile python -c "
  import torch
  a = torch.randn(4096, 4096, device='cuda')
  b = torch.randn(4096, 4096, device='cuda')
  c = torch.matmul(a, b)
  torch.cuda.synchronize()
  "
  # 打开结果
  nsys-ui my_profile.nsys-rep
  ```
- [ ] 在时间线中识别：
  - CUDA kernel 执行时间
  - 内存拷贝时间（H2D / D2H）
  - API 调用开销

**做完之后能了解**：
- Nsight Systems 是看"时间线"的工具（回答"什么时候发生了什么"）
- Nsight Compute 是看"kernel 详情"的工具（回答"这个 kernel 为什么慢"）

---

### Day 5：NUMA 与 CPU 亲和性

- [ ] 阅读 Chapter 3 中 "NUMA Awareness and CPU Pinning"
- [ ] 理解 NUMA：Non-Uniform Memory Access，不同 CPU socket 访问不同内存区域速度不同
- [ ] **本地实验**：检查你的机器是否有 NUMA
  ```bash
  numactl --hardware
  lscpu | grep NUMA
  ```
- [ ] 理解 CPU Pinning：把进程绑定到特定 CPU 核心，避免跨 NUMA 访问
  ```bash
  # 绑定到 CPU 0-7
  numactl --cpunodebind=0 --membind=0 python train.py
  ```
- [ ] 思考：为什么 "不绑 CPU" 可能导致 20-30% 的性能损失？

**做完之后能了解**：
- NUMA 对多 GPU 训练的影响
- 为什么 CPU 亲和性对 DataLoader 性能很重要

---

## 第二部分：CUDA 编程（Day 6-12）

### Day 6：CUDA 线程层次

- [ ] 精读 Chapter 6 "GPU Architecture, CUDA Programming, and Maximizing Occupancy"
- [ ] 理解 CUDA 线程层次：
  - **Grid**：整个 kernel 的所有 block
  - **Block**：一组线程，可以共享 Shared Memory
  - **Thread**：最基本的执行单元
  - **Warp**：32 个线程，同一 warp 内的线程执行 SIMT
- [ ] **本地实验**：写第一个 CUDA kernel
  ```python
  import torch

  # PyTorch 自动编译 CUDA kernel
  @torch.jit.script
  def add_kernel(a, b):
      return a + b

  a = torch.randn(1000000, device='cuda')
  b = torch.randn(1000000, device='cuda')
  c = add_kernel(a, b)
  print(c[:5])
  ```
- [ ] 用 Triton 写更灵活的 kernel：
  ```python
  import triton
  import triton.language as tl

  @triton.jit
  def add_kernel_triton(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(axis=0)
      block_start = pid * BLOCK_SIZE
      offsets = block_start + tl.arange(0, BLOCK_SIZE)
      mask = offsets < n_elements
      x = tl.load(x_ptr + offsets, mask=mask)
      y = tl.load(y_ptr + offsets, mask=mask)
      tl.store(output_ptr + offsets, x + y, mask=mask)

  def add(x, y):
      output = torch.empty_like(x)
      n_elements = output.numel()
      grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
      add_kernel_triton[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
      return output

  a = torch.randn(1000000, device='cuda')
  b = torch.randn(1000000, device='cuda')
  c = add(a, b)
  print(c[:5])
  ```

**做完之后能了解**：
- CUDA kernel 不是"在 GPU 上跑 Python"，而是"在 GPU 上并行执行同一个函数"
- Triton 比手写 CUDA C++ 更易用，但概念相同

---

### Day 7：内存访问模式

- [ ] 精读 Chapter 7 "Profiling and Tuning GPU Memory Access Patterns"
- [ ] 理解关键概念：
  - **Coalesced Access**：相邻线程访问相邻内存地址（最优）
  - **Uncoalesced Access**：相邻线程访问分散内存地址（性能差）
  - **Bank Conflict**：多个线程同时访问 Shared Memory 的同一个 bank
- [ ] **本地实验**：对比 coalesced 和 uncoalesced 的性能
  ```python
  import torch
  import time

  device = 'cuda'
  n = 4096 * 4096

  # Coalesced: 顺序访问
  x = torch.randn(n, device=device)
  torch.cuda.synchronize()
  t0 = time.time()
  y = x[::1]  # 顺序
  torch.cuda.synchronize()
  print(f"Coalesced: {(time.time()-t0)*1000:.2f} ms")

  # Uncoalesced: 跨步访问
  torch.cuda.synchronize()
  t0 = time.time()
  y = x[::16]  # 跨 16 个元素
  torch.cuda.synchronize()
  print(f"Uncoalesced: {(time.time()-t0)*1000:.2f} ms")
  ```
- [ ] 观察性能差异（通常 5-10 倍）

**做完之后能了解**：
- 为什么内存访问模式比计算量更重要
- "计算快但内存访问乱序" = 性能差

---

### Day 8：Shared Memory 与 Tiling

- [ ] 精读 Chapter 7 的 "Tiling and Data Reuse Using Shared Memory"
- [ ] 理解 Tiling：把大数据分块加载到 Shared Memory，在块内复用
- [ ] 理解为什么 Shared Memory 比全局内存快 ~100 倍：
  - 全局内存：离 SM 远，带宽有限
  - Shared Memory：在 SM 内部，片上存储
- [ ] **本地实验**：用 PyTorch 模拟 tiling 的效果
  ```python
  import torch

  # 矩阵乘法 C = A @ B
  # 不 tiling：直接乘
  A = torch.randn(1024, 1024, device='cuda')
  B = torch.randn(1024, 1024, device='cuda')
  C = torch.matmul(A, B)  # cuBLAS 内部自动 tiling

  # tiling 的核心思想：
  # 把 A 分成 32x32 的块，B 分成 32x32 的块
  # 每次加载一块到 Shared Memory，计算部分结果，累加
  ```
- [ ] 阅读 PyTorch 的 `torch.matmul` 源码或 FlashAttention 的实现，观察 tiling 策略

**做完之后能了解**：
- Tiling 如何让内存访问从 O(N³) 降到 O(N²)（在 Shared Memory 内复用数据）
- 为什么 FlashAttention 的核心优化就是 tiling（分块计算 attention，避免全局内存访问）

---

### Day 9：Occupancy 与 Warp 效率

- [ ] 精读 Chapter 8 "Occupancy Tuning, Warp Efficiency, and Instruction-Level Parallelism"
- [ ] 理解 Occupancy：每个 SM 上同时活跃的 warp 数量 / 理论最大值
- [ ] 理解影响 Occupancy 的因素：
  - 寄存器使用量（用多了，warp 数量就少了）
  - Shared Memory 使用量（用多了，warp 数量就少了）
  - Block 大小（通常 128-256 线程最优）
- [ ] 理解 Warp Divergence：同一 warp 内的线程走不同分支
  ```python
  # 假设有 32 个线程（一个 warp）
  # 线程 0-15：走 if 分支
  # 线程 16-31：走 else 分支
  # 实际执行：先执行 if 分支（线程 16-31 等待），再执行 else 分支（线程 0-15 等待）
  # 效率 = 50%
  ```
- [ ] **本地实验**：用 Nsight Compute 分析一个 kernel 的 occupancy
  ```bash
  ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_elapsed \
      python -c "import torch; a=torch.randn(4096,4096,device='cuda'); b=torch.randn(4096,4096,device='cuda'); c=torch.matmul(a,b)"
  ```

**做完之后能了解**：
- 为什么 occupancy 不是越高越好（高 occupancy 可能意味着每个 warp 的寄存器变少）
- Warp divergence 是性能杀手的原理

---

### Day 10：PyTorch 编译器优化

- [ ] 精读 Chapter 13-14 "Profiling, Tuning, and Scaling PyTorch" 和 "PyTorch Compiler"
- [ ] 理解 `torch.compile` 的三种模式：
  - `default`：平衡编译时间和运行速度
  - `reduce-overhead`：最小化 Python 开销
  - `max-autotune`：最大化性能（编译时间最长）
- [ ] **本地实验**：对比 `torch.compile` 前后的性能
  ```python
  import torch
  import time

  def my_model(x):
      for _ in range(10):
          x = torch.nn.functional.gelu(x @ torch.randn(4096, 4096, device='cuda'))
      return x

  x = torch.randn(256, 4096, device='cuda')

  # 不编译
  torch.cuda.synchronize()
  t0 = time.time()
  for _ in range(100):
      y = my_model(x)
  torch.cuda.synchronize()
  print(f"Eager: {(time.time()-t0)*1000:.2f} ms")

  # 编译
  compiled_model = torch.compile(my_model, mode="max-autotune")
  compiled_model(x)  # 第一次编译
  torch.cuda.synchronize()
  t0 = time.time()
  for _ in range(100):
      y = compiled_model(x)
  torch.cuda.synchronize()
  print(f"Compiled: {(time.time()-t0)*1000:.2f} ms")
  ```
- [ ] 观察编译后性能提升（通常 1.5-2 倍）

**做完之后能了解**：
- `torch.compile` 把 Python 代码编译为优化的 Triton/CUDA kernel
- 为什么第一次调用慢（编译时间），后续调用快

---

### Day 11：OpenAI Triton 编程

- [ ] 精读 Chapter 14 的 "Writing Custom Kernels with OpenAI Triton"
- [ ] 理解 Triton 的抽象层次：
  - CUDA C++：手动管理 block、warp、register、shared memory
  - Triton：自动 tiling、内存合并、shared memory 管理
- [ ] **本地实验**：用 Triton 实现一个 Softmax kernel
  ```python
  import triton
  import triton.language as tl
  import torch

  @triton.jit
  def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
      row_idx = tl.program_id(0)
      col_offsets = tl.arange(0, BLOCK_SIZE)
      mask = col_offsets < n_cols

      input_ptrs = input_ptr + row_idx * n_cols + col_offsets
      row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

      row_minus_max = row - tl.max(row, axis=0)
      numerator = tl.exp(row_minus_max)
      denominator = tl.sum(numerator, axis=0)
      softmax_output = numerator / denominator

      output_ptrs = output_ptr + row_idx * n_cols + col_offsets
      tl.store(output_ptrs, softmax_output, mask=mask)

  def softmax(x):
      n_rows, n_cols = x.shape
      BLOCK_SIZE = triton.next_power_of_2(n_cols)
      y = torch.empty_like(x)
      softmax_kernel[(n_rows,)](x, y, n_cols, BLOCK_SIZE=BLOCK_SIZE)
      return y

  x = torch.randn(128, 256, device='cuda')
  y_triton = softmax(x)
  y_torch = torch.softmax(x, dim=1)
  print(torch.allclose(y_triton, y_torch, atol=1e-5))  # True
  ```
- [ ] 对比 Triton Softmax 和 PyTorch `torch.softmax` 的性能

**做完之后能了解**：
- Triton 如何用 Python 级别的代码编写高性能 GPU kernel
- Triton 自动处理 tiling 和内存合并

---

### Day 12：CUDA 编程周复盘

- [ ] 回答：
  - 为什么 Shared Memory 比全局内存快 ~100 倍？
  - Tiling 的核心思想是什么？
  - Occupancy 受哪些因素限制？
  - `torch.compile` 为什么能加速？
  - Triton 相比 CUDA C++ 的优势和劣势？
- [ ] 记录：你写过的所有 kernel，标注各自的性能瓶颈

---

## 第三部分：推理优化（Day 13-15）

### Day 13：推理并行策略

- [ ] 精读 Chapter 15-17 "Multi-Node Inference Parallelism"
- [ ] 理解三种推理并行：
  - **Tensor Parallel (TP)**：把一层切分到多卡，适合低延迟
  - **Pipeline Parallel (PP)**：把不同层分配到多卡，适合大模型
  - **Expert Parallel (EP)**：MoE 模型的 expert 分布在多卡
- [ ] 理解 **Disaggregated Prefill/Decode**：
  - Prefill：计算密集型，需要高算力
  - Decode：内存密集型，需要大显存带宽
  - 分离后可以用不同硬件分别优化
- [ ] **本地实验**：用 SGLang 启动一个小模型，对比单卡和多卡（云租 2×A100）的推理延迟

**做完之后能了解**：
- 为什么 "Prefill 和 Decode 分离" 是大模型推理的趋势
- 不同并行策略对延迟和吞吐的影响

---

### Day 14：KV Cache 优化

- [ ] 精读 Chapter 18 "Advanced Prefill-Decode and KV Cache Tuning"
- [ ] 理解 KV Cache 的内存占用：
  - 每层：2 × num_heads × head_dim × seq_len × batch_size × bytes_per_element
  - 总内存 = 层数 × 每层内存
- [ ] 理解优化策略：
  - **FlashAttention**：分块计算，减少全局内存访问
  - **PagedAttention**：分页管理 KV Cache，减少碎片
  - **KV Cache 量化**：INT8/FP8 压缩
  - **KV Cache Offload**：把不常用的 KV 放到 CPU 内存
- [ ] **本地实验**：估算一个 7B 模型在不同 seq_len 下的 KV Cache 大小
  ```python
  layers = 32
  num_heads = 32
  head_dim = 128
  batch_size = 1
  seq_len = 4096
  bytes_per_element = 2  # FP16

  kv_cache_per_layer = 2 * num_heads * head_dim * seq_len * batch_size * bytes_per_element
  total_kv_cache = layers * kv_cache_per_layer
  print(f"KV Cache: {total_kv_cache / 1024**3:.2f} GB")
  ```

**做完之后能了解**：
- KV Cache 是大模型推理的显存瓶颈（不是模型权重）
- 为什么长上下文（100K+ tokens）推理需要优化 KV Cache

---

### Day 15：性能检查清单

- [ ] 阅读本书的 "200+ Item Performance Checklist"
- [ ] 选出与你最相关的 20 项：
  - GPU 编程相关：_____
  - PyTorch 调优相关：_____
  - 分布式训练相关：_____
  - 推理优化相关：_____
- [ ] **本地实验**：用检查清单逐项检查你的环境
  ```bash
  # 检查 GPU 驱动版本
  nvidia-smi
  # 检查 CUDA 版本
  nvcc --version
  # 检查 PyTorch 是否用到了 CUDA
  python -c "import torch; print(torch.cuda.is_available())"
  # 检查 NCCL 版本
  python -c "import torch; print(torch.cuda.nccl.version())"
  ```

**做完之后能了解**：
- 系统性的性能优化不是"调几个参数"，而是"逐项检查清单"
- 为什么 "驱动版本不匹配" 可能导致 30% 的性能损失

---

## 第四部分：全局复盘（Day 16-17）

### Day 16：综合实验

- [ ] 设计一个完整的优化实验：
  1. 选一个基准任务（如矩阵乘法或注意力计算）
  2. 用 Nsight 分析瓶颈
  3. 应用一种优化（tiling、shared memory、kernel fusion）
  4. 验证性能提升
- [ ] 记录：优化前后的性能对比、瓶颈分析、优化思路

---

### Day 17：全书总结

- [ ] 回答：
  - Roofline 模型中，如何判断程序是 compute-bound 还是 memory-bound？
  - 为什么 Tiling 能显著减少内存访问？
  - `torch.compile` 的三种模式各适用于什么场景？
  - Prefill/Decode 分离的好处是什么？
  - KV Cache 量化的精度损失如何控制？
- [ ] 记录：这本书对你最有价值的 3 个知识点
- [ ] 记录：你还想深入了解但没覆盖到的 3 个主题

---

## 附录：核心概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| Roofline Model | 用"峰值算力"和"峰值带宽"判断程序瓶颈 |
| Coalesced Access | 相邻线程访问相邻内存地址（最优访问模式）|
| Tiling | 把大数据分块加载到 Shared Memory 复用 |
| Occupancy | SM 上活跃的 warp 比例 |
| Warp Divergence | 同一 warp 内线程走不同分支 |
| torch.compile | 把 PyTorch 代码编译为优化 CUDA kernel |
| Triton | Python 级别的 GPU kernel 编程语言 |
| Disaggregated Prefill/Decode | 把 prefill 和 decode 分配到不同 GPU |
| PagedAttention | 分页管理 KV Cache，减少碎片 |
| KV Cache Quantization | 用 INT8/FP8 压缩 KV Cache |

---

## 附录：本地可复现实验清单

| 实验 | 命令/代码 | 预期结果 |
|------|-----------|----------|
| 监控 GPU 状态 | `nvidia-smi dmon -s umt` | 实时显示利用率、显存带宽、温度 |
| Roofline 分析 | 计算算术强度 | 判断 compute-bound / memory-bound |
| Nsight 录制 | `nsys profile -o out python script.py` | 生成时间线报告 |
| 矩阵乘法对比 | `torch.matmul(a, b)` vs `a @ b` | cuBLAS 自动优化 |
| torch.compile | 编译前后对比 | 通常 1.5-2 倍加速 |
| Triton Softmax | 自定义 kernel | 与 `torch.softmax` 精度一致 |
| KV Cache 估算 | 公式计算 | 明确显存占用 |
