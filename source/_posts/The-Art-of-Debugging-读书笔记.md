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

## 核心理念

这本书的目标是教你：

1. **快速调试正常问题**
2. **让复杂问题变得可调试**

> （改编自 Perl 的口号："Easy things should be easy and hard things should be possible"）

## 概述

这是一本持续演进中的调试方法论和可复制粘贴的配方集合，用于成功调试简单和复杂的软件问题。部分章节已经相当完整，有些将在后续阶段完成，还有一些尚未开始。

除了调试方法论外，本书的第二个重点是分享作者发现的最佳工具，以便成功完成调试工作，同时理想地"在此过程中少掉几根头发"。

作者 Stas Bekman 自 1995 年起就开始从事软件开发，其中大量工作涉及调试。多年来，他发展了各种高效的方法论来发现问题的根源——这是解决问题之前最困难的阶段。因为一旦问题被理解，通常解决方案就触手可及。

## 主要内容

### 1. Fast Debugging Methodology（快速调试方法论）

高效的调试流程和思维框架，帮助快速定位问题根源。

### 2. Debugging Compiled Programs（调试编译程序）

涉及的工具和概念包括：
- `gdb` — GNU 调试器
- `ldd` — 查看动态链接依赖
- `nm` — 列出目标文件中的符号
- `LD_LIBRARY_PATH` — 动态链接库搜索路径
- `LD_PRELOAD` — 预加载共享库

### 3. Debugging Python（调试 Python）

- `py-spy` — 无侵入式 Python 性能分析器
- Python 路径问题排查
- 自动打印调试技巧

### 4. Debugging PyTorch（调试 PyTorch）

- CPU 和 GPU 内存问题
- 性能瓶颈分析
- 模型调试
- 张量相关问题

### 5. Unix Tools For Debugging（Unix 调试工具）

- `bash` 脚本调试
- `strace` — 系统调用追踪
- `make` 构建调试
- Shell prompt 调试技巧
- `nohup` 后台任务管理

### 6. Debugging Machine Learning Projects（调试机器学习项目）

外部资源集合，针对 ML 项目特有的调试场景。

## 写作背景

作者提到，在调试过程中，同事时常建议他将这些方法分享给世界。虽然他一直认为这些方法很难泛化，但最近这个"埋下的种子"似乎发芽了。因此，在接下来的章节中，他将尝试分享一些见解，以缓解这个有时非常困难的过程。

由于作者没有保存用例，在真空中写调试内容非常困难，因此需要一段时间来积累这些内容。预计这些页面将在很长一段时间内保持"进行中"（WIP）状态。但希望一些想法能尽早传达给读者，并帮助减轻他们在专业项目和业余项目中的调试负担。

## 许可证

Attribution-ShareAlike 4.0 International
