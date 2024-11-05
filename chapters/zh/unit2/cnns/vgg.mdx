# 用于大规模图像识别的超深卷积网络（2014）

## 简介

VGG架构由牛津大学视觉几何组的Karen Simonyan和Andrew Zisserman于2014年开发，因此被命名为VGG。该模型在当时显著优于之前的模型，特别是在2014年的Imagenet挑战赛中，也称为ILSVRC。

## VGG网络架构

- 输入为224x224的图像。
- 卷积核的形状为(3,3)，最大池化窗口的形状为(2,2)。
- 每个卷积层的通道数量为64 -> 128 -> 256 -> 512 -> 512。
- VGG16具有16个隐藏层（13个卷积层和3个全连接层）。
- VGG19具有19个隐藏层（16个卷积层和3个全连接层）。

## 关键比较

- VGG（16或19层）相比当时的其他SOTA网络相对更深。例如，2012年ILSVRC的获胜模型AlexNet仅有8层。
- 采用多个小（3x3）感受野的卷积滤波器并配合ReLU激活函数，而不是一个大的（7x7或11x11）滤波器，能够更好地学习复杂特征。较小的滤波器还意味着每层的参数更少，同时在层间引入了额外的非线性。
- 多尺度训练和推理。每张图像在多个不同尺度下训练多轮，以确保不同大小的图像具有相似的特征。
- VGG网络的一致性和简洁性使其更易于扩展或进行未来改进。

## PyTorch示例

下面是VGG19的PyTorch实现代码。

```python
import torch.nn as nn


class VGG19(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG19, self).__init__()

        # 特征提取层：卷积和池化层
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                3, 64, kernel_size=3, padding=1
            ),  # 3个输入通道，64个输出通道，3x3卷积核，1像素填充
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2, stride=2
            ),  # 2x2卷积核，步长为2的最大池化
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 池化层
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        # 用于分类的全连接层
        self.classifier = nn.Sequential(
            nn.Linear(
                512 * 7 * 7, 4096
            ),  # 512通道，最大池化后的空间维度为7x7
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout层，概率为0.5
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),  # 输出层，具有'num_classes'个输出单元
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # 通过特征提取层
        x = self.avgpool(x)  # 通过池化层
        x = x.view(x.size(0), -1)  # 将输出展平以传递到全连接层
        x = self.classifier(x)  # 通过分类层
        return x
```