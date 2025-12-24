# SHARP 项目 UV 快速安装指南

本指南提供了使用 UV 包管理器快速安装和设置 SHARP 项目的详细步骤。UV 是一个高性能的 Python 包管理器，能够加速依赖安装过程。

## 项目介绍

SHARP 是一个用于从单张图像生成逼真视图合成的方法。它通过神经网络的单次前向传递，在不到一秒的时间内回归出场景的 3D 高斯表示，然后可以实时渲染附近视图的高分辨率逼真图像。

## 前提条件

- 操作系统：Windows、Linux 或 macOS
- Python 3.13（建议使用）
- CUDA GPU（用于渲染轨迹，预测功能支持 CPU、CUDA 和 MPS）

## 1. 安装 UV

### Windows

在 PowerShell 中运行以下命令：

```powershell
(Invoke-WebRequest -Uri https://astral.sh/uv/install.ps1 -UseBasicParsing).Content | PowerShell -
```

### Linux/macOS

在终端中运行以下命令：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. 克隆仓库

```bash
git clone https://github.com/apple/ml-sharp.git
cd ml-sharp
```

## 3. 创建虚拟环境

使用 UV 创建一个 Python 3.13 虚拟环境：

```bash
uv venv --python 3.13 venv
```

## 4. 激活虚拟环境

### Windows

```powershell
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

## 5. 安装依赖

使用 UV 安装项目依赖：

```bash
uv pip install -r requirements.txt
```

## 6. 验证安装

运行以下命令验证安装是否成功：

```bash
sharp --help
```

如果安装成功，您将看到类似以下输出：

```
Usage: sharp [OPTIONS] COMMAND [ARGS]...

  Run inference for SHARP model.

Options:
  --help  Show this message and exit.

Commands:
  predict  Predict Gaussians from input images.
  render   Predict Gaussians from input images.
```

## 使用示例

### 预测 3D 高斯表示

```bash
sharp predict -i /path/to/input/images -o /path/to/output/gaussians
```

### 渲染轨迹视频（需要 CUDA GPU）

```bash
# 从输入图像预测并渲染
sharp predict -i /path/to/input/images -o /path/to/output/gaussians --render

# 从已有的高斯表示渲染
sharp render -i /path/to/output/gaussians -o /path/to/output/renderings
```

## 故障排除

### UV 安装失败

- 确保您的网络连接正常
- 尝试使用管理员权限运行安装命令
- 检查系统防火墙设置，确保允许 UV 下载依赖

### 依赖安装失败

- 确保您使用的是推荐的 Python 3.13 版本
- 尝试清理 UV 缓存：`uv cache clean`
- 检查 `requirements.txt` 文件是否存在且完整

### 运行 `sharp --help` 失败

- 确保虚拟环境已正确激活
- 检查是否所有依赖都已成功安装
- 尝试重新安装项目：`uv pip install -e .`

## 相关链接

- [项目主页](https://apple.github.io/ml-sharp/)
- [UV 官方文档](https://docs.astral.sh/uv/)
- [SHARP 论文](https://arxiv.org/abs/2512.10685)

## 许可证

请查看仓库中的 [LICENSE](LICENSE) 和 [LICENSE_MODEL](LICENSE_MODEL) 文件了解许可证信息。