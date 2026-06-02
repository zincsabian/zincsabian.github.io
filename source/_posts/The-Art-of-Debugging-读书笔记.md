---
title: The Art of Debugging 读书笔记（逐日展开版）
date: 2026-06-01 14:07:00
categories: 读书笔记
tags:
  - Debug
  - 工程实践
  - Python
---

> 原书地址：https://github.com/stas00/the-art-of-debugging
>
> 作者：Stas Bekman（1995 年起从事软件开发）
> 许可：CC BY-SA 4.0
>
> 这是一本关于调试方法论和工具集的开放书籍。本文按"先学方法论 → 再练工具 → 最后实战"的路径逐日展开，每天配一个可复现的调试案例。

<!-- more -->

---

## 写在前面

### 调试是工程能力的核心

编程时间分配：
- 写代码：20%
- 读代码：30%
- **调试：40%**
- 开会：10%

如果你每天工作 8 小时，**超过 3 小时在调试**。把调试效率提高一倍，等于每天多出 1.5 小时。

### 你的环境

这本书的所有工具和方法都在你的 RTX 5060 + AMD 9600X 上完全可复现。不需要大模型，不需要多卡。

---

## 第一部分：调试方法论（Day 1-5）

### Day 1：调试的两个核心需求

- [ ] 阅读本书 "Fast Debugging Methodology" 章节
- [ ] 理解两个核心需求：
  1. **快速迭代**：从重启到关键断点的等待时间应控制在**几秒**
  2. **小数据**：用最小数据复现问题
- [ ] **本地实验**：故意写一个有 bug 的程序，分别用大模型和小模型来调试
  ```python
  # 有 bug 的函数：应该返回 a + b，但写成了 a * b
  def add(a, b):
      return a * b  # 故意写错

  # 大数据调试：痛苦
  import torch
  big_a = torch.randn(10000, 10000)
  big_b = torch.randn(10000, 10000)
  result = add(big_a, big_b)  # 等 30 秒，结果还是错的，不知道哪里错了

  # 小数据调试：清晰
  small_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
  small_b = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
  result = add(small_a, small_b)
  print(result)
  # 预期：[[2, 3], [4, 5]]
  # 实际：[[1, 2], [3, 4]]  # 明显是乘法不是加法！
  ```
- [ ] 对比两种调试方式的效率差异

**做完之后能了解**：
- 为什么 "用大数据调试" 是痛苦且低效的
- "小数据 + 快速迭代" 是调试的万能钥匙

---

### Day 2：数据选择策略

- [ ] 阅读 "Real data vs. random data vs. synthetic data" 章节
- [ ] 理解三种数据的选择策略：
  - **Random data**：程序崩溃时用（不需要真实数据）
  - **Synthetic data**：需要跟踪数值时用（如 `[[1.0, 2.0], [3.0, 4.0]]`）
  - **Real data**：程序能跑但结果不对时用（需要真实数据验证质量）
- [ ] **本地实验**：用 synthetic data 调试一个数值计算 bug
  ```python
  # 假设你在实现一个矩阵乘法
  def my_matmul(a, b):
      # 故意有一个 bug：忘记转置
      return a @ b.T  # 应该是 a @ b

  # 用 synthetic data：容易发现错误
  a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
  b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # 单位矩阵
  result = my_matmul(a, b)
  print(result)
  # 预期：a @ I = a = [[1, 2], [3, 4]]
  # 实际：因为 .T 的存在，结果是 [[1, 3], [2, 4]]
  # 一眼看出错误！
  ```
- [ ] 记录：你过去一周调试时，用的是哪种数据？有没有可以改进的地方？

**做完之后能了解**：
- 不同调试阶段需要不同类型的数据
- synthetic data 的选择技巧：用容易心算的数字、用 stand-out 的值（如 `DEADBEEF`）

---

### Day 3：调试流程标准化

- [ ] 阅读 "Fast Debugging Methodology" 中关于调试流程的部分
- [ ] 理解标准调试流程：
  ```
  1. 复现问题 → 确保能稳定复现
  2. 最小化 → 用最少的代码和数据复现
  3. 假设 → 提出可能的原因
  4. 验证 → 设计实验验证假设
  5. 修复 → 确认根因后修复
  6. 回归测试 → 确保修复没有引入新问题
  ```
- [ ] **本地实验**：用标准流程解决一个 bug
  ```python
  # 步骤 1：复现
  def buggy_function(x):
      return x / (x - 5)  # 当 x=5 时崩溃

  buggy_function(5)  # ZeroDivisionError

  # 步骤 2：最小化
  # 已经是最小化了，一行代码

  # 步骤 3：假设
  # 假设：分母为 0 时没处理

  # 步骤 4：验证
  # 验证：x=5 时，x-5=0，确实会 ZeroDivisionError

  # 步骤 5：修复
  def fixed_function(x):
      if x == 5:
          return float('inf')
      return x / (x - 5)

  # 步骤 6：回归测试
  assert fixed_function(5) == float('inf')
  assert fixed_function(6) == 6.0
  assert fixed_function(4) == -4.0
  print("All tests passed!")
  ```

**做完之后能了解**：
- 调试不是"碰运气"，而是可以系统化的六步法
- 为什么 "先最小化再调试" 能节省 80% 的时间

---

### Day 4：常见调试陷阱

- [ ] 阅读书中关于调试陷阱的部分
- [ ] 列出 5 个常见陷阱：
  1. **确认偏误**：只看支持自己假设的证据
  2. **过度复杂化**：把简单问题想复杂
  3. **忽略日志**：不仔细读错误信息就急着改代码
  4. **没有版本控制**：改了代码后忘记原来是什么样子
  5. **假设环境没问题**："在我机器上能跑" 的陷阱
- [ ] **本地实验**：故意犯一个确认偏误的错误
  ```python
  # 假设你认为 bug 是因为学习率太高
  # 于是你只调学习率，但问题其实是梯度爆炸
  # 你应该同时检查：学习率、梯度范数、loss 曲线、权重分布
  ```
- [ ] 理解：为什么 "看错误信息" 能解决 50% 的 bug（大多数错误信息已经告诉你问题在哪）

**做完之后能了解**：
- 调试中最常见的认知偏误
- 如何避免 "在错误的方向上越走越远"

---

### Day 5：方法论周复盘

- [ ] 回答：
  - 调试的两个核心需求是什么？
  - 什么时候用 random data？什么时候用 synthetic data？什么时候用 real data？
  - 标准调试流程的 6 个步骤是什么？
  - 你最容易犯哪个调试陷阱？
- [ ] 记录：过去一周的调试经历，用标准流程重新分析一次

---

## 第二部分：Python 调试工具（Day 6-10）

### Day 6：q —— 快速打印追踪

- [ ] 阅读 "Debugging Python" 章节的 `q` 工具部分
- [ ] 安装 `q`：`pip install q`
- [ ] **本地实验**：用 `q` 追踪一个函数的输入输出
  ```python
  import q

  @q
  def add(a, b):
      return a + b

  @q
  def multiply(a, b):
      return a * b

  result = add(1, 2)
  result = multiply(result, 3)

  # 查看 /tmp/q 的输出
  # cat /tmp/q
  # 输出：
  # 0.0s add(1, 2)
  # 0.0s -> 3
  # 0.0s multiply(3, 3)
  # 0.0s -> 9
  ```
- [ ] 理解 `q` 的优势：
  - 输出到 `/tmp/q`，不干扰正常日志
  - 自动打印参数和返回值
  - 自带粗略的时间戳

**做完之后能了解**：
- 为什么 `q` 比 `print` 更适合调试（不污染 stdout，自动格式化）
- 在已有大量日志输出的程序中，如何快速定位自己的调试信息

---

### Day 7：pdb 与 ipdb —— 断点调试

- [ ] 阅读书中关于 Python 调试的部分
- [ ] 掌握 `pdb` 的基本命令：
  - `b(reak) line_no`：设置断点
  - `c(ontinue)`：继续执行
  - `n(ext)`：执行下一行
  - `s(tep)`：进入函数
  - `r(eturn)`：执行到函数返回
  - `p(rint) var`：打印变量
  - `l(ist)`：显示当前代码
  - `q(uit)`：退出
- [ ] **本地实验**：用 `pdb` 调试一个循环
  ```python
  import pdb

  def find_max(numbers):
      max_val = numbers[0]
      for i, num in enumerate(numbers):
          if num > max_val:
              max_val = num
          if i == 3:  # 在第四次迭代时打断点
              pdb.set_trace()
      return max_val

  result = find_max([3, 1, 4, 1, 5, 9, 2, 6])
  print(result)

  # 在 pdb 中输入：
  # p max_val  # 查看当前最大值
  # p i        # 查看当前索引
  # p num      # 查看当前数字
  # n          # 执行下一行
  # c          # 继续执行
  ```
- [ ] 对比 `pdb` 和 `ipdb`（`ipdb` 支持语法高亮和 tab 补全）

**做完之后能了解**：
- 断点调试比 "加 print" 更高效（可以查看任意变量、任意时刻的状态）
- `pdb` 是 Python 标准库的调试器，无需安装

---

### Day 8：py-spy —— 无侵入式性能分析

- [ ] 阅读 "Debugging Python" 章节的 `py-spy` 部分
- [ ] 安装 `py-spy`：`pip install py-spy`
- [ ] **本地实验**：用 `py-spy` 分析一个慢程序
  ```python
  # slow_program.py
  import time
  import torch

  def slow_function():
      time.sleep(1)  # 模拟慢操作
      x = torch.randn(1000, 1000)
      y = torch.matmul(x, x)  # 计算密集型
      return y

  def fast_function():
      return 42

  for i in range(10):
      if i % 2 == 0:
          slow_function()
      else:
          fast_function()
  ```
  ```bash
  # 终端 1：运行程序
  python slow_program.py

  # 终端 2：用 py-spy 分析
  py-spy top --pid <pid>

  # 录制火焰图
  py-spy record -o profile.svg --pid <pid>
  # 用浏览器打开 profile.svg，观察时间分布
  ```
- [ ] 观察火焰图，确认 `slow_function` 占用了大部分时间

**做完之后能了解**：
- `py-spy` 不需要修改代码、不需要重启程序
- 火焰图能直观显示时间消耗在哪些函数上
- 为什么 "无侵入式" 很重要（生产环境不能改代码）

---

### Day 9：objprint 与打印对象结构

- [ ] 阅读 "Printing Object Variables" 部分
- [ ] 理解 `__repr__` 的作用：让对象打印更友好
- [ ] **本地实验**：对比有 `__repr__` 和没有的类
  ```python
  # 没有 __repr__
  class BadModel:
      def __init__(self):
          self.layers = [1, 2, 3]
          self.activation = 'relu'

  # 有 __repr__
  class GoodModel:
      def __init__(self):
          self.layers = [1, 2, 3]
          self.activation = 'relu'
      def __repr__(self):
          return f"GoodModel(layers={self.layers}, activation='{self.activation}')"

  print(BadModel())   # <__main__.BadModel object at 0x...>
  print(GoodModel())  # GoodModel(layers=[1, 2, 3], activation='relu')
  ```
- [ ] 安装 `objprint`：`pip install objprint`
- [ ] 用 `objprint` 打印复杂对象的结构
  ```python
  from objprint import objprint
  import torch

  model = torch.nn.Linear(10, 10)
  objprint(model)  # 打印模型的完整结构
  ```

**做完之后能了解**：
- 为什么调试时 "看到对象的内部结构" 很重要
- `__repr__` 和 `__str__` 的区别（`__repr__` 是给开发者看的，`__str__` 是给用户看的）

---

### Day 10：Python 调试周复盘

- [ ] 回答：
  - `q` 和 `print` 的区别是什么？
  - `pdb` 的 `n` 和 `s` 有什么区别？
  - `py-spy` 为什么被称为 "无侵入式"？
  - 什么时候需要自定义 `__repr__`？
- [ ] 记录：过去一周的 Python 调试中，你最常用的工具是什么？有没有可以改进的地方？

---

## 第三部分：PyTorch 调试（Day 11-15）

### Day 11：检测 NaN/Inf

- [ ] 阅读 "Debugging PyTorch" 章节
- [ ] 掌握检测方法：
  - `torch.isnan(tensor).any()`：检测 NaN
  - `torch.isinf(tensor).any()`：检测 Inf
  - `torch.autograd.set_detect_anomaly(True)`：自动检测
- [ ] **本地实验**：故意制造 NaN，然后定位来源
  ```python
  import torch

  # 开启异常检测
  torch.autograd.set_detect_anomaly(True)

  x = torch.tensor([1.0], requires_grad=True)
  y = x.log()  # log(1) = 0，正常
  z = y / 0    # 除以 0！

  try:
      z.backward()
  except RuntimeError as e:
      print(f"Caught error: {e}")
      # 会打印出 forward stack trace，告诉你哪一步出了问题
  ```
- [ ] 理解 `detect_anomaly` 的工作原理：在反向传播时检查梯度是否为 NaN/Inf

**做完之后能了解**：
- PyTorch 的 `detect_anomaly` 是定位 NaN 来源的神器
- 为什么 NaN 一旦产生就会传播（任何运算 involving NaN 都是 NaN）

---

### Day 12：显存分析

- [ ] 阅读书中关于显存调试的部分
- [ ] 掌握显存分析工具：
  ```python
  import torch

  # 查看当前显存占用
  print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
  print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
  print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

  # 重置峰值统计
  torch.cuda.reset_peak_memory_stats()

  # 显存摘要
  print(torch.cuda.memory_summary())
  ```
- [ ] **本地实验**：追踪一次 forward 的显存变化
  ```python
  import torch

  torch.cuda.reset_peak_memory_stats()
  before = torch.cuda.memory_allocated()

  model = torch.nn.Linear(4096, 4096).cuda()
  x = torch.randn(256, 4096, device='cuda')
  y = model(x)

  after = torch.cuda.memory_allocated()
  print(f"Model: {(after - before) / 1024**2:.2f} MB")
  print(f"Peak: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
  ```
- [ ] 理解 `memory_allocated` 和 `memory_reserved` 的区别：
  - `allocated`：PyTorch 实际使用的显存
  - `reserved`：CUDA 缓存池中的显存（包括已分配和未分配的空闲块）

**做完之后能了解**：
- 如何系统性地追踪显存泄漏
- 为什么 `reserved > allocated`（CUDA 内存池机制）

---

### Day 13：PyTorch Profiler

- [ ] 阅读书中关于 PyTorch Profiler 的部分
- [ ] **本地实验**：用 PyTorch Profiler 分析一个模型
  ```python
  import torch
  from torch.profiler import profile, record_function, ProfilerActivity

  model = torch.nn.TransformerEncoder(
      torch.nn.TransformerEncoderLayer(d_model=512, nhead=8),
      num_layers=6
  ).cuda()
  x = torch.randn(32, 100, 512, device='cuda')

  with profile(
      activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
      record_shapes=True,
      profile_memory=True,
      with_stack=True
  ) as prof:
      with record_function("model_forward"):
          y = model(x)

  # 打印结果
  print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

  # 导出 Chrome trace
  prof.export_chrome_trace("trace.json")
  # 打开 chrome://tracing，加载 trace.json
  ```
- [ ] 观察 Chrome trace，找到耗时最长的 kernel

**做完之后能了解**：
- PyTorch Profiler 是定位性能瓶颈的标准工具
- Chrome trace 能直观显示 CPU 和 GPU 的时间线

---

### Day 14：梯度检查

- [ ] 理解梯度检查的重要性
- [ ] **本地实验**：检查一个模型的梯度
  ```python
  import torch

  model = torch.nn.Sequential(
      torch.nn.Linear(10, 10),
      torch.nn.ReLU(),
      torch.nn.Linear(10, 1)
  ).cuda()

  x = torch.randn(32, 10, device='cuda')
  y = torch.randn(32, 1, device='cuda')

  loss_fn = torch.nn.MSELoss()
  pred = model(x)
  loss = loss_fn(pred, y)
  loss.backward()

  # 检查每层梯度
  for name, param in model.named_parameters():
      if param.grad is not None:
          grad_norm = param.grad.norm().item()
          print(f"{name}: grad_norm={grad_norm:.4f}")
          if torch.isnan(param.grad).any():
              print(f"  WARNING: NaN in gradient!")
          if grad_norm < 1e-6:
              print(f"  WARNING: Vanishing gradient!")
  ```
- [ ] 理解梯度消失和梯度爆炸的表现

**做完之后能了解**：
- 为什么 "模型不收敛" 时首先要检查梯度
- 梯度范数的正常范围（通常 0.1-10，太小说明梯度消失，太大说明梯度爆炸）

---

### Day 15：PyTorch 调试周复盘

- [ ] 回答：
  - `torch.autograd.set_detect_anomaly(True)` 的原理是什么？
  - `memory_allocated` 和 `memory_reserved` 的区别？
  - PyTorch Profiler 的 Chrome trace 怎么看？
  - 梯度范数多少算正常？多少算异常？
- [ ] 记录：过去一周的 PyTorch 调试中，哪个工具帮你解决了实际问题？

---

## 第四部分：Unix 调试工具（Day 16-19）

### Day 16：strace —— 系统调用追踪

- [ ] 阅读 "Unix Tools For Debugging" 章节的 `strace` 部分
- [ ] **本地实验**：用 `strace` 追踪一个 Python 程序的系统调用
  ```bash
  # 追踪所有系统调用
  strace -e trace=file python -c "import torch; torch.randn(10)" 2>&1 | head -20

  # 只看文件操作
  strace -e trace=file,open,read,write python your_script.py 2>&1 | grep -E "open|read|write"

  # 统计每种系统调用的次数
  strace -c python your_script.py 2>&1
  ```
- [ ] 理解输出格式：
  ```
  open("/path/to/file", O_RDONLY) = 3  # 打开文件，返回文件描述符 3
  read(3, "data...", 1024) = 1024       # 从 fd 3 读取 1024 字节
  close(3) = 0                          # 关闭 fd 3
  ```

**做完之后能了解**：
- `strace` 是诊断 "程序为什么慢" 的利器（可能是卡在 I/O）
- 如何区分 "CPU 计算慢" 和 "I/O 等待慢"

---

### Day 17：htop / nvtop —— 资源监控

- [ ] 阅读书中关于系统监控的部分
- [ ] **本地实验**：同时运行 `htop` 和 `nvtop`，观察资源使用
  ```bash
  # 终端 1
  htop

  # 终端 2
  nvtop

  # 终端 3：跑一个压力测试
  python -c "import torch; a=torch.randn(10000,10000,device='cuda'); [torch.matmul(a,a) for _ in range(100)]"
  ```
- [ ] 观察：
  - CPU 利用率是否 100%？（如果是，可能是 DataLoader 瓶颈）
  - GPU 利用率是否 100%？（如果不是，可能是 CPU 等待或 I/O 瓶颈）
  - 显存占用是否稳步上升？（可能是泄漏）

**做完之后能了解**：
- `htop` 看 CPU/内存，`nvtop` 看 GPU/显存
- "GPU 利用率 100%" 不一定代表高效（可能是在等内存）

---

### Day 18：tmux + nohup —— 后台与持久化

- [ ] 阅读书中关于进程管理的部分
- [ ] **本地实验**：
  ```bash
  # 用 tmux 创建会话
  tmux new -s train
  # 在 tmux 中运行训练
  python train.py
  # 按 Ctrl+B 然后 D 分离会话

  # 重新连接
  tmux attach -t train

  # 用 nohup 后台运行
  nohup python train.py > train.log 2>&1 &
  # 查看输出
  tail -f train.log
  ```
- [ ] 理解 `tmux` 和 `nohup` 的适用场景：
  - `tmux`：需要交互式操作（随时 attach 进去看）
  - `nohup`：纯后台运行（适合长时间训练）

**做完之后能了解**：
- 为什么 "SSH 断开训练不中断" 是刚需
- `tmux` 是远程服务器上最实用的工具之一

---

### Day 19：Unix 调试周复盘 + 全书总结

- [ ] 回答：
  - `strace` 能解决什么问题？不能解决什么问题？
  - `htop` 中 "CPU 利用率 100% 但程序很慢" 可能是什么原因？
  - `tmux` 和 `nohup` 各适用于什么场景？
  - 调试方法论的两个核心需求是什么？
  - Python 调试的四个工具（q/pdb/py-spy/objprint）各适用于什么场景？
  - PyTorch 调试的四个工具（detect_anomaly/profiler/memory/gradient）各适用于什么场景？
- [ ] 记录：这本书对你最有价值的 3 个知识点
- [ ] 记录：你还想深入了解但没覆盖到的 3 个主题

---

## 附录：调试工具速查表

| 工具 | 用途 | 安装 | 适用场景 |
|------|------|------|----------|
| `q` | 快速打印追踪 | `pip install q` | 不污染 stdout 的调试 |
| `pdb` | 断点调试 | 内置 | 逐行调试 |
| `ipdb` | 增强版 pdb | `pip install ipdb` | 需要语法高亮和补全 |
| `py-spy` | 性能分析 | `pip install py-spy` | 无侵入式 profiling |
| `objprint` | 打印对象结构 | `pip install objprint` | 查看复杂对象 |
| `strace` | 系统调用追踪 | 系统自带 | 诊断 I/O 问题 |
| `htop` | CPU/内存监控 | `apt install htop` | 资源使用监控 |
| `nvtop` | GPU 监控 | `apt install nvtop` | 显存/GPU 监控 |
| `tmux` | 会话管理 | `apt install tmux` | 持久化终端会话 |
| `nohup` | 后台运行 | 内置 | 长时间后台任务 |

---

## 附录：调试决策树

```
程序崩溃了？
├── 是 → 能否稳定复现？
│   ├── 否 → 检查随机种子、并发问题
│   └── 是 → 用最小数据复现
│       ├── 用 q/pdb 定位崩溃点
│       └── 用 detect_anomaly 检查 NaN
└── 否 → 程序能跑但结果不对？
    ├── 是 → 检查输入数据
    │   ├── 数据正确？→ 检查模型输出
    │   │   ├── 输出正确？→ 检查后处理
    │   │   └── 输出错误？→ 检查中间层
    │   │       ├── 用 py-spy 找性能瓶颈
    │   │       └── 用 Profiler 分析时间线
    │   └── 数据错误？→ 修复数据预处理
    └── 否 → 程序太慢？
        ├── 用 py-spy 定位热点
        ├── 用 nvtop 看 GPU 利用率
        ├── 用 strace 检查 I/O
        └── 用 Profiler 分析 CUDA kernel
```

---

## 附录：本地可复现调试案例

| 案例 | 现象 | 诊断 | 解决 |
|------|------|------|------|
| NaN Loss | loss 变成 nan | `detect_anomaly` + 梯度检查 | 梯度裁剪、降低学习率 |
| OOM | CUDA out of memory | `memory_summary()` | 减小 batch、用 gradient checkpoint |
| 慢训练 | 100% GPU 但吞吐低 | `py-spy` + Profiler | 检查 DataLoader、启用 compile |
| 死锁 | 程序卡住不动 | `strace -p <pid>` | 检查分布式通信 |
| 结果不一致 | 每次运行结果不同 | 检查随机种子 | 设置 `torch.manual_seed` |
