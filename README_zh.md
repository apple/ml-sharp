# SHARP：不到一秒的单目视图合成

[![项目页面](https://img.shields.io/badge/Project-Page-green)](https://apple.github.io/ml-sharp/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.10685-b31b1b.svg)](https://arxiv.org/abs/2512.10685)

本软件项目是研究论文《Sharp Monocular View Synthesis in Less Than a Second》的配套代码，作者为：_Lars Mescheder, Wei Dong, Shiwei Li, Xuyang Bai, Marcel Santos, Peiyun Hu, Bruno Lecouat, Mingmin Zhen, Amaël Delaunoy, Tian Fang, Yanghai Tsin, Stephan Richter and Vladlen Koltun_。

![](data/teaser.jpg)

我们提出了SHARP，一种从单张图像生成逼真视图合成的方法。给定一张照片，SHARP通过神经网络的单次前向传递，在标准GPU上不到一秒的时间内回归出所描绘场景的3D高斯表示。然后，可以实时渲染SHARP生成的3D高斯表示，为附近视图生成高分辨率逼真图像。该表示是度量的，具有绝对尺度，支持度量相机运动。实验结果表明，SHARP在多个数据集上表现出稳健的零样本泛化能力。与最佳现有模型相比，它在多个数据集上建立了新的技术水平，将LPIPS降低了25–34%，DISTS降低了21–43%，同时将合成时间降低了三个数量级。

## 快速开始

我们建议首先创建一个Python环境：

```
conda create -n sharp python=3.13
```

然后，您可以使用以下命令安装项目：

```
pip install -r requirements.txt
```

要测试安装是否成功，运行：

```
sharp --help
```

## 使用命令行界面

要运行预测：

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians
```

模型检查点将在首次运行时自动下载并缓存在本地路径 `~/.cache/torch/hub/checkpoints/`。

或者，您可以直接下载模型：

```
wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt
```

要使用手动下载的检查点，请使用 `-c` 参数指定：

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians -c sharp_2572gikvuh.pt
```

结果将是输出文件夹中的3D高斯图元（3DGS）。3DGS `.ply` 文件兼容各种公共3DGS渲染器。我们遵循OpenCV坐标约定（x向右，y向下，z向前）。3DGS场景中心大致在 (0, 0, +z)。使用第三方渲染器时，请相应地缩放和旋转以重新定位场景中心。

### 渲染轨迹（仅支持CUDA GPU）

此外，您还可以使用相机轨迹渲染视频。虽然高斯预测适用于所有CPU、CUDA和MPS，但通过 `--render` 选项渲染视频目前需要CUDA GPU。gsplat渲染器在首次启动时需要一段时间初始化。

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians --render

# 或者从中间高斯结果渲染：
sharp render -i /path/to/output/gaussians -o /path/to/output/renderings
```

## 评估

有关定量和定性评估，请参阅论文。此外，您可以查看此[定性示例页面](https://apple.github.io/ml-sharp/)，其中包含与相关工作的多个视频比较。

## 引用

如果您发现我们的工作有用，请引用以下论文：

```bibtex
@inproceedings{Sharp2025:arxiv,
  title      = {Sharp Monocular View Synthesis in Less Than a Second},
  author     = {Lars Mescheder and Wei Dong and Shiwei Li and Xuyang Bai and Marcel Santos and Peiyun Hu and Bruno Lecouat and Mingmin Zhen and Ama"{e}l Delaunoy and Tian Fang and Yanghai Tsin and Stephan R. Richter and Vladlen Koltun},
  journal    = {arXiv preprint arXiv:2512.10685},
  year       = {2025},
  url        = {https://arxiv.org/abs/2512.10685},
}
```

## 致谢

我们的代码库基于多个开源贡献构建，请查看[ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS)了解更多详细信息。

## 许可证

在使用提供的代码之前，请查看仓库中的[LICENSE](LICENSE)文件，
对于发布的模型，请查看[LICENSE_MODEL](LICENSE_MODEL)文件。