---
title: SGLang Day 1 — 源码编译、启动与调试完全记录
date: 2026-06-02 11:00:00
categories: SGLang
tags:
  - SGLang
  - 源码启动
  - RTX 5060
---

> 本文记录 SGLang 从源码安装、编译到启动推理服务的完整过程，适用于 **RTX 5060 8GB** 本地开发环境。
> 对应主规划：[SGLang 结构学习完全规划（基于 RTX 5060 8GB + 云租分布式）](/2026/06/02/SGLang-结构学习完全规划/)

---

## 一、环境前提

- CUDA Toolkit 已安装（建议 12.4+，5060 需要 CUDA 12.x）
- Python 3.10+
- 8GB 显存，启动时必须加显存控制参数

---

## 二、开发者标准安装流程

### 1. 确认源码位置

```bash
cd ~/devroot/sglang          # 你的源码目录
git checkout $(git describe --tags --abbrev=0)  # 切到最新 release
```

### 2. 安装 Python 依赖（editable 模式）

```bash
cd ~/devroot/sglang
source .venv-study/bin/activate

# 开发模式安装 sglang + 所有 CUDA 依赖
pip install -e "python/[cuda]"
```

> `pip install -e` 是 **editable install**（软链接），修改 `python/sglang/` 下的代码会**立即生效**，无需重复安装。

### 3. 编译 C++/CUDA 内核（sgl-kernel）

```bash
cd ~/devroot/sglang/sgl-kernel
pip install -e .
```

> 这个步骤编译 `sglang-kernel` 包（即 Python 代码里 `import sgl_kernel` 的来源），**必须完成**，否则启动时会报 `ModuleNotFoundError: No module named 'sgl_kernel'`。
> 编译时间约 5-15 分钟，需要 nvcc。

---

## 三、直接用源码启动（不依赖 pip install）

有时候你不想通过 pip 入口启动，而是**直接跑当前目录下的源码**（例如你同时在改代码，想确认加载的是当前目录版本）：

### 方式 A：PYTHONPATH + 模块启动（最灵活）

```bash
cd ~/devroot/sglang
source .venv-study/bin/activate

PYTHONPATH=python python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --tp-size 1 \
  --mem-fraction-static 0.85
```

**原理**：`PYTHONPATH=python` 把 `~/devroot/sglang/python/` 加入 Python 模块搜索路径，解释器直接加载源码目录的 `sglang/` 包。

### 方式 B：走 CLI 入口（支持模型类型自动检测）

```bash
cd ~/devroot/sglang
source .venv-study/bin/activate

PYTHONPATH=python python -m sglang.cli.main serve \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --tp-size 1 \
  --mem-fraction-static 0.85
```

**原理**：`sglang.cli.main:main` 是 `sglang serve` 命令的真正实现，支持自动检测 LLM / Diffusion 模型类型。

### 方式 C：直接运行 launch_server.py（旧入口）

```bash
cd ~/devroot/sglang/python
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --tp-size 1 \
  --mem-fraction-static 0.85
```

> ⚠️ 这种方式会发出 `UserWarning`，提示 `sglang serve` 是推荐入口。

---

## 四、RTX 5060 8GB 专用启动命令

8GB 显存必须选小模型，推荐以下配置：

### 最稳妥：1.5B 模型（~4GB 显存）

```bash
PYTHONPATH=python python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --tp-size 1 \
  --mem-fraction-static 0.85
```

### 进阶：3B 模型（~6GB 显存）

```bash
PYTHONPATH=python python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --tp-size 1 \
  --mem-fraction-static 0.85
```

### 极限：Phi-4-mini 3.8B（~7GB 显存）

```bash
PYTHONPATH=python python -m sglang.launch_server \
  --model-path microsoft/Phi-4-mini-instruct \
  --tp-size 1 \
  --mem-fraction-static 0.85
```

> `--mem-fraction-static 0.85` 是 **5060 必须参数**，只让 SGLang 使用 85% 显存（约 6.8GB），给 CUDA 上下文和系统预留 1GB+，避免初始化 OOM。

---

## 五、验证服务启动

```bash
# 另开终端
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'
```

---

## 六、开发者调试方式

### 1. 修改代码立即生效

因为 `pip install -e` 是 editable，直接改 `python/sglang/srt/` 下的文件即可：

```bash
# 示例：在 scheduler.py 加一行日志
vim python/sglang/srt/managers/scheduler.py
# 加一行：logger.info("=== Scheduler step ===")

# 重新启动，日志立即生效
python -m sglang.launch_server --model-path ...
```

### 2. 用 py-spy 观察运行时线程

```bash
# 找到 sglang 进程 PID
ps aux | grep sglang

# 实时查看 Python 线程状态
py-spy top --pid <pid>

# 录制火焰图
py-spy record -o profile.svg --pid <pid>
```

### 3. 确保加载的是当前目录源码

如果你同时装了 pip 版和源码版，用 `PYTHONPATH=python` 强制加载当前目录：

```bash
PYTHONPATH=python python -c "import sglang; print(sglang.__file__)"
# 应输出 ~/devroot/sglang/python/sglang/__init__.py
```

### 4. 进程树观察

```bash
# 启动后观察有几个进程
ps aux | grep sglang

# 典型进程结构：
# - python -m sglang.launch_server      (主进程)
#   ├─ python ... tokenizer_manager      (Tokenizer 子进程)
#   ├─ python ... scheduler              (Scheduler 子进程，每卡一个)
#   └─ python ... detokenizer_manager    (Detokenizer 子进程)
```

---

## 七、常见问题

### Q1：`python -m sglang.launch_server` 报 `No module named 'sgl_kernel'`

**原因**：`sgl-kernel` 没编译安装。
**解决**：
```bash
cd ~/devroot/sglang/sgl-kernel && pip install -e .
```

### Q2：`python -m sglang.launch_server` 报 `No module named 'flashinfer_python'`

**原因**：Python 依赖没装好。
**解决**：
```bash
cd ~/devroot/sglang
pip install -e "python/[cuda]"
```

### Q3：启动时 CUDA OOM，但 `nvidia-smi` 显示显存还有余量

**原因**：5060 的 8GB 在 CUDA 初始化时会额外分配上下文内存。
**解决**：始终加 `--mem-fraction-static 0.85`，或换更小的模型。

### Q4：改了代码但没生效

**原因**：可能加载了 pip 缓存里的 sglang，而不是源码目录。
**解决**：用 `PYTHONPATH=python` 强制加载当前目录。

---

## 八、关键路径速查

| 目的 | 文件/命令 |
|------|----------|
| 旧版启动入口 | `python/sglang/launch_server.py` |
| CLI 启动入口 | `python/sglang/cli/main.py` → `sglang.cli.serve` |
| HTTP 服务实现 | `python/sglang/srt/entrypoints/http_server.py` |
| Engine 子进程启动 | `python/sglang/srt/entrypoints/engine.py` |
| 内核编译目录 | `sgl-kernel/`（CMake + scikit-build-core） |
| Python 依赖声明 | `python/pyproject.toml` |
| 启动参数解析 | `python/sglang/srt/server_args.py` |

---

> **总结**：开发者学习 SGLang 的核心流程是 `pip install -e`（editable）+ `sgl-kernel` 编译，之后用 `python -m sglang.launch_server` 或 `PYTHONPATH=python python -m sglang.launch_server` 启动。5060 用户务必控制显存并选择 1.5B-3B 小模型。
