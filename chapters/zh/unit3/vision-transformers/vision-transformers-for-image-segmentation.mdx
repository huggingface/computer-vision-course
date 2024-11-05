# 基于 Transformer 的图像分割

在本节中，我们将探讨 Vision Transformers 在图像分割方面与卷积神经网络 (CNN) 的比较，并以一个基于 Vision Transformer 的分割模型架构作为示例进行详细讲解。

<Tip warning={true}>
  本节假设您熟悉图像分割、卷积神经网络 (CNN) 和 Vision Transformers 的基础知识。如果您对这些概念不熟悉，建议先学习课程中相关的基础材料。
</Tip>

## CNN 与 Transformer 在分割任务中的对比

在 Vision Transformers 出现之前，卷积神经网络 (CNN) 是图像分割的首选。例如，[U-Net](https://arxiv.org/abs/1505.04597) 和 [Mask R-CNN](https://arxiv.org/abs/1703.06870) 等模型能够捕捉区分图像中不同对象所需的细节，使其在分割任务中表现出色。

尽管 CNN 模型在过去十年中取得了优异的结果，但它们仍存在一些限制，Transformer 的出现则意在解决这些问题：

- **空间局限性**：CNN 通过小的感受野学习局部模式，这种局部关注使它们难以“连接”图像中相隔较远但相关的特征，从而影响其准确分割复杂场景或对象的能力。与 CNN 不同，ViT 利用注意力机制设计用于捕捉图像中的全局依赖关系。这意味着基于 ViT 的模型能够一次性考虑整个图像，从而理解图像中远距离部分之间的复杂关系。对于分割任务，这种全局视角能够更准确地划分对象。
- **任务特定组件**：如 Mask R-CNN 之类的方法包含手工设计的组件（如非最大值抑制、空间锚点）以编码有关分割任务的先验知识。这些组件增加了复杂性并需要手动调整。而基于 ViT 的分割方法通过消除手工设计的组件，简化了分割过程，更易于优化。
- **分割任务的专业化**：基于 CNN 的分割模型分别处理语义、实例和全景分割任务，从而为每个任务设计了专门的架构，且每个任务都需要单独的研究。近期的基于 ViT 的模型如 [MaskFormer](https://arxiv.org/abs/2107.06278)、[SegFormer](https://arxiv.org/abs/2105.15203) 或 [SAM](https://arxiv.org/abs/2304.02643) 提供了一种统一的方法，在一个框架内处理语义、实例和全景分割任务。

## MaskFormer 聚焦：阐释 ViT 在图像分割中的应用

MaskFormer ([论文](https://arxiv.org/abs/2107.06278)，[Hugging Face transformers 文档](https://huggingface.co/docs/transformers/en/model_doc/maskformer)) 在论文《MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation》中提出，是一种为图像中每个类预测分割掩码的模型，将语义分割和实例分割统一在一个架构中。

### MaskFormer 架构

下图展示了论文中的架构图。

<img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/maskformer_architecture.png"/>

该架构由以下三部分组成：

**像素级模块**：利用主干网络提取图像特征，并通过像素解码器生成逐像素嵌入。

**Transformer 模块**：使用标准 Transformer 解码器从图像特征和可学习的位置嵌入（查询）中计算每个分割的嵌入，编码每个分割的全局信息。

**分割模块**：使用线性分类器和多层感知机 (MLP) 分别生成每个分割的类别概率预测和掩码嵌入。掩码嵌入结合逐像素嵌入，用于预测每个分割的二进制掩码。

该模型使用与 [DETR](https://huggingface.co/learn/computer-vision-course/unit3/vision-transformers/detr) 相同的二进制掩码损失以及每个预测分割的交叉熵分类损失进行训练。

### Hugging Face Transformers 实现的全景分割推理示例

全景分割任务是对图像中的每个像素进行分类，同时识别这些类别中的不同对象，结合了语义分割和实例分割。

```python
from transformers import pipeline
from PIL import Image
import requests

segmentation = pipeline("image-segmentation", "facebook/maskformer-swin-base-coco")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

results = segmentation(images=image, subtask="panoptic")
results
```

如以下示例所示，结果包含同一类别的多个实例，每个实例都有不同的掩码。

```bash
[
  {
    "score": 0.993197,
    "label": "remote",
    "mask": <PIL.Image.Image image mode=L size=640x480 at 0x109363910>
  },
  {
    "score": 0.997852,
    "label": "cat",
    "mask": <PIL.Image.Image image mode=L size=640x480 at 0x1093635B0>
  },
  {
    "score": 0.998006,
    "label": "remote",
    "mask": <PIL.Image.Image image mode=L size=640x480 at 0x17EE84670>
  },
  {
    "score": 0.997469,
    "label": "cat",
    "mask": <PIL.Image.Image image mode=L size=640x480 at 0x17EE87100>
  }
]
```

## 微调基于 Vision Transformer 的分割模型

随着许多预训练分割模型的出现，迁移学习和微调已被广泛应用于将这些模型适应于特定的用例，特别是因为基于 Transformer 的分割模型（如 MaskFormer）对数据需求量大，从零开始训练难度较大。这些技术利用预训练表示来有效地将这些模型适应新的数据。通常，对于 MaskFormer，会保持主干网络、像素解码器和 Transformer 解码器冻结，以利用它们所学到的通用特征，而 Transformer 模块则会微调，以适应新的分割任务的类别预测和掩码生成需求。

[本教程笔记本](https://colab.research.google.com/github/johko/computer-vision-course/blob/main/notebooks/Unit%203%20-%20Vision%20Transformers/transfer-learning-segmentation.ipynb)将带您通过 MaskFormer 进行图像分割的迁移学习示例。

## 参考文献

- [MaskFormer Hugging Face 文档](https://huggingface.co/docs/transformers/en/model_doc/maskformer)
- [图像分割 Hugging Face 任务指南](https://huggingface.co/docs/transformers/en/tasks/semantic_segmentation)