---
title: The Art of Debugging 读书笔记
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

本书是一本持续演进中的调试方法论和可复制粘贴的配方集合，目标是教你**快速调试正常问题**，并让**复杂问题变得可调试**。以下内容按原书结构整理，包含大量可直接使用的代码片段和工具推荐。

<!-- more -->

---

## 一、快速调试方法论

### 1.1 两个核心需求

成功调试只需要满足两个条件：

1. **快速迭代（Quick Iterations）**
   - 从重启程序到到达关键断点的时间应控制在**几秒**内
   - 如果需要等 10 分钟，重复 30 次，人会疲劳、会放弃，还会忘记之前的尝试
   - **任何缩短循环时间的改进都会提高调试成功率**

2. **小数据（Small Payload）**
   - 用小甚至 tiny 的数据调试
   - 原因：
     - 快速重启（配合上面的需求）
     - 容易记住、比较和心算

```python
# 容易调试的数据
tensor([[0.1, 0.2],
        [0.3, 0.4]])

# 难以调试的数据
tensor([[0.8860, 0.7966, 0.0274,  ..., 0.4142, 0.7156, 0.3564],
        [0.3885, 0.1056, 0.3069,  ..., 0.8970, 0.8329, 0.0012],
        ...])
```

### 1.2 数据选择策略

**程序崩溃时：**
- 不需要真实数据，任何随机或合成数据都可以

**程序能跑但结果不对时：**
- 需要真实数据

**数据渐进策略：**
1. 先用随机/合成数据验证基本功能
2. 再用小规模真实数据验证质量
3. 最后用完整数据做最终验证

**ML 场景的数据规模建议：**
- 功能调试：10K 参数模型
- 基本质量测试：1B 或 125M 参数模型
- 最终训练：175B 参数模型

**合成数据的技巧：**
- 使用容易记住的数字：`[[1.0, 2.0], [3.0, 4.0]]`
- 使用 standout 的值：汇编调试中的 `DEADBEEF`

### 1.3 核心原则总结

> 先用最小的数据让程序跑通，再逐步扩大规模。

---

## 二、调试编译程序

涉及的核心工具和概念：

### 2.1 gdb（GNU Debugger）

- 设置断点：`break main`
- 单步执行：`step`、`next`
- 查看变量：`print var`
- 查看调用栈：`backtrace`
- 条件断点：`break file.c:10 if x > 5`

### 2.2 ldd

- 查看可执行文件的动态链接依赖
- 检查是否缺少共享库

```bash
ldd ./my_program
```

### 2.3 nm

- 列出目标文件中的符号表
- 检查函数/变量是否存在

```bash
nm -C my_program | grep my_function
```

### 2.4 LD_LIBRARY_PATH

- 指定动态链接库的搜索路径
- 常用于切换不同版本的库

```bash
export LD_LIBRARY_PATH=/path/to/custom/libs:$LD_LIBRARY_PATH
```

### 2.5 LD_PRELOAD

- 预加载指定的共享库，覆盖原始函数
- 常用于拦截系统调用、注入调试代码

```bash
LD_PRELOAD=./my_hook.so ./my_program
```

---

## 三、调试 Python 程序

### 3.1 q —— 快速打印追踪工具

`q` 包（`pip install q`）专为快速代码和函数追踪设计，输出到 `/tmp/q` 文件，不干扰正常的 stdout/stderr。

```python
import q

# 追踪任意表达式
q(1 + 2)

# 追踪函数（自动记录参数和返回值）
@q
def add(a, b):
    return a + b

add(1, 2)
```

查看输出：
```bash
tail -F /tmp/q
# 输出：
# 0.0s <module>: 1+2=3
# 0.0s add(1, 2)
# 0.0s -> 3
```

**MacOS 注意：** 使用前需要设置：
```bash
export TMPDIR=/tmp/
```

### 3.2 打印对象变量

默认 `print(obj)` 只输出对象地址，不够友好：

```python
class A:
    foo = 1
    def __init__(self):
        self.bar = 2

a = A()
print(a)  # <__main__.A object at 0x...>
```

添加 `__repr__` 方法：

```python
class A:
    foo = 1
    def __init__(self):
        self.bar = 2
    def __repr__(self):
        return f"A(foo={self.foo}, bar={self.bar})"

a = A()
print(a)  # A(foo=1, bar=2)
```

### 3.3 py-spy

- 无侵入式 Python 性能分析器
- 不需要修改代码、不需要重启程序
- 可以实时查看火焰图

```bash
# 实时查看 Top 函数
py-spy top --pid <pid>

# 生成火焰图
py-spy record -o profile.svg --pid <pid>
```

### 3.4 自动打印调试

在函数入口处自动打印参数：

```python
import functools

def debug_print(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@debug_print
def my_function(x, y):
    return x + y
```

---

## 四、调试 PyTorch 程序

### 4.1 CPU/GPU 内存问题

**查看 GPU 内存使用：**
```python
import torch

print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

**检测内存泄漏：**
```python
import torch

before = torch.cuda.memory_allocated()
# 运行你的代码
after = torch.cuda.memory_allocated()
print(f"Memory delta: {(after - before) / 1e6:.2f} MB")
```

### 4.2 性能瓶颈分析

**PyTorch Profiler：**
```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("model_inference"):
        model(input)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 4.3 检测 NaN/Inf

```python
# 启用异常检测（自动在出现 NaN/Inf 时抛出异常）
torch.autograd.set_detect_anomaly(True)

# 手动检查张量
if torch.isnan(tensor).any():
    print("NaN detected!")
if torch.isinf(tensor).any():
    print("Inf detected!")
```

### 4.4 模型调试技巧

**打印模型结构：**
```python
print(model)
```

**查看每层输出形状：**
```python
from torchsummary import summary
summary(model, input_size=(3, 224, 224))
```

**检查梯度：**
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

---

## 五、Unix 调试工具

### 5.1 strace —— 系统调用追踪

追踪程序的所有系统调用：

```bash
# 追踪所有系统调用
strace ./my_program

# 只追踪文件操作
strace -e trace=file ./my_program

# 追踪网络操作
strace -e trace=network ./my_program

# 输出到文件
strace -o trace.log ./my_program
```

### 5.2 nohup —— 后台任务管理

让程序在后台持续运行，即使终端断开：

```bash
nohup python train.py > output.log 2>&1 &
# 查看任务
tail -f output.log
```

### 5.3 make 调试

```bash
# 只打印命令，不执行
make -n

# 详细模式
make V=1

# 调试模式
make -d
```

### 5.4 Shell Prompt 调试技巧

在 PS1 中显示当前 Git 分支、Python 虚拟环境等信息：

```bash
# 显示 Git 分支
export PS1='\u@\h:\w$(__git_ps1 " (%s)")\$ '

# 显示虚拟环境
export PS1='(${VIRTUAL_ENV##*/}) '$PS1
```

### 5.5 快速创建最小复现环境

```bash
# 创建隔离的测试目录
mkdir /tmp/debug_test && cd /tmp/debug_test

# 使用最小的数据集
python -c "import torch; torch.save(torch.randn(10, 10), 'tiny_data.pt')"

# 运行测试
python my_script.py --data tiny_data.pt
```

---

## 六、调试方法论总结

### 6.1 调试流程

1. **复现问题** —— 确保能稳定复现
2. **最小化** —— 用最少的代码和数据复现
3. **假设** —— 提出可能的原因假设
4. **验证** —— 设计实验验证假设
5. **修复** —— 确认根因后修复
6. **回归测试** —— 确保修复没有引入新问题

### 6.2 常见陷阱

- **确认偏误**：只看支持自己假设的证据
- **过度复杂化**：把简单问题想复杂
- **忽略日志**：不仔细读错误信息就急着改代码
- **没有版本控制**：改了代码后忘记原来是什么样子

### 6.3 黄金法则

> **理解问题比修复问题更重要。**
>
> 一旦真正理解了 bug 的原因，修复通常是轻而易举的。

---

## 七、推荐工具清单

| 工具 | 用途 | 安装 |
|------|------|------|
| `q` | Python 快速打印追踪 | `pip install q` |
| `py-spy` | 无侵入式 Python profiler | `pip install py-spy` |
| `gdb` | C/C++ 调试器 | 系统包管理器 |
| `strace` | 系统调用追踪 | 系统包管理器 |
| `nvidia-smi` | GPU 状态监控 | NVIDIA 驱动自带 |
| `htop` | 交互式进程查看器 | 系统包管理器 |
| `ncdu` | 磁盘使用情况分析 | 系统包管理器 |
