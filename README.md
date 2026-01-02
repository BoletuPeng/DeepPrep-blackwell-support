# DeepPrep Blackwell (RTX 50 Series) Support Guide

[English](#english) | [中文](#中文)

---

## English

### What is this?

This guide helps you run [DeepPrep](https://github.com/pBFSLab/DeepPrep) on **NVIDIA RTX 50 series GPUs** (RTX 5090, 5080, 5070, etc.) which use the **Blackwell architecture (sm_120)**.

### The Problem

The official DeepPrep Docker image (`pbfslab/deepprep:25.1.0`) uses PyTorch 2.0.1+cu118, which doesn't support Blackwell architecture. When you try to run it, you'll see:

```
NVIDIA GeForce RTX 5090 Laptop GPU with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.

RuntimeError: CUDA error: no kernel image is available for execution on the device
```

### The Solution

This guide walks you through upgrading PyTorch to 2.7.0+cu128 and resolving all dependency conflicts, including:

- Upgrading PyTorch from 2.0.1+cu118 to 2.7.0+cu128
- Recompiling PyTorch3D from source
- Updating torch-scatter, torch-sparse, torch-cluster, torch-spline-conv, and pyg-lib
- Converting SUGAR model files for torch_geometric 2.7.0 compatibility

### Quick Start

📖 **[Read the full installation guide](./INSTALL_GUIDE_EN.md)**

### Requirements

- Docker Desktop installed and working
- Original DeepPrep image pulled (`pbfslab/deepprep:25.1.0`)
- ~100 GB free disk space
- NVIDIA RTX 50 series GPU with latest drivers

### Tested Environment

| Component | Version |
|-----------|---------|
| DeepPrep | 25.1.0 |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU |
| Host OS | Windows 11 |
| Docker | Docker Desktop for Windows |
| Final PyTorch | 2.7.0+cu128 |
| Final CUDA | 12.8 |

### Contributing

Issues and pull requests are welcome! If you've tested this on other Blackwell GPUs or found improvements, please share.

### License

MIT License - See [LICENSE](./LICENSE)

### Acknowledgments

- [DeepPrep](https://github.com/pBFSLab/DeepPrep) by pBFSLab
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d) by Meta Research

---

## 中文

### 这是什么？

本指南帮助您在 **NVIDIA RTX 50 系列显卡**（RTX 5090、5080、5070 等）上运行 [DeepPrep](https://github.com/pBFSLab/DeepPrep)。这些显卡使用 **Blackwell 架构 (sm_120)**。

### 问题描述

DeepPrep 官方 Docker 镜像 (`pbfslab/deepprep:25.1.0`) 使用的 PyTorch 2.0.1+cu118 不支持 Blackwell 架构。运行时会出现以下错误：

```
NVIDIA GeForce RTX 5090 Laptop GPU with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.

RuntimeError: CUDA error: no kernel image is available for execution on the device
```

### 解决方案

本指南详细介绍如何将 PyTorch 升级到 2.7.0+cu128 并解决所有依赖冲突，包括：

- 将 PyTorch 从 2.0.1+cu118 升级到 2.7.0+cu128
- 从源码重新编译 PyTorch3D
- 更新 torch-scatter、torch-sparse、torch-cluster、torch-spline-conv 和 pyg-lib
- 转换 SUGAR 模型文件以兼容 torch_geometric 2.7.0

### 快速开始

📖 **[阅读完整安装指南](./INSTALL_GUIDE.md)**

### 系统要求

- 已安装并正常运行的 Docker Desktop
- 已拉取原版 DeepPrep 镜像 (`pbfslab/deepprep:25.1.0`)
- 约 100 GB 可用磁盘空间
- NVIDIA RTX 50 系列显卡，并安装最新驱动

### 测试环境

| 组件 | 版本 |
|------|------|
| DeepPrep | 25.1.0 |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU |
| 主机系统 | Windows 11 |
| Docker | Docker Desktop for Windows |
| 最终 PyTorch | 2.7.0+cu128 |
| 最终 CUDA | 12.8 |

### 贡献

欢迎提交 Issue 和 Pull Request！如果您在其他 Blackwell 显卡上测试过，或发现了改进方法，请分享。

### 许可证

MIT License - 见 [LICENSE](./LICENSE)

### 致谢

- [DeepPrep](https://github.com/pBFSLab/DeepPrep) by pBFSLab
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d) by Meta Research
