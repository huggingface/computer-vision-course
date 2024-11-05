# 介绍

在上一单元中，我们学习了多模态，尤其是如何融合视觉和语言模型，以利用两者的优点，并在零样本图像分类等任务中超越简单的视觉模型。
另一个多模态模型有显著影响的领域是生成视觉模型。在本单元中，我们将更深入地探讨这些类型的神经网络。

## 定义

什么是生成视觉模型？它们如何与其他模型不同？

数学模型通常可以分为两大类：生成模型和判别模型。
判别模型和生成模型的主要区别在于，判别模型学习不同类别的边界，而生成模型学习不同类别的分布。

判别模型可以应用于标准的计算机视觉任务，如分类和回归，这些任务可以扩展为更复杂的过程，如语义分割或目标检测。

为简洁起见，在本章中，我们将讨论解决以下任务的生成模型：

* 噪声到图像（DCGAN）
* 文本到图像（扩散模型）
* 图像到图像（StyleGAN, CycleGAN, 扩散模型）

本节将涵盖两种生成模型：基于GAN的模型和基于扩散的模型。

## 计算机视觉中生成模型的评估

通常来说，为生成模型设计有意义的评估指标是非常困难的。因为通常没有一个可靠的“真实值”，并且很难量化图像的质量。FID 是最常用的指标，但它并不完美。

我们快速了解一下 FID。FID 代表 Fréchet Inception Distance，它是对 Inception Score 的改进，并在 [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/pdf/1706.08500.pdf) 中提出。FID 被认为对噪声和生成图像中可能存在的某些伪影具有抗性。FID 越低越好。

FID 的计算方式是通过从 Inception-v3 特征中构建两个分布。第一个分布是从训练数据特征计算的，第二个分布是从生成图像特征计算的。然后计算这两个分布之间的 Fréchet 距离，这就是你的 FID 分数。分数越低，生成图像的感知质量越高。这里有一个关于 FID 的[简短解释](https://www.youtube.com/watch?v=9zTwSzXxNDo&t=398s)。

其他可能遇到的指标包括 SSIM, PSNR, IS（Inception Score）以及最近引入的 CLIP Score。

* PSNR（峰值信噪比）几乎可以解释为均方误差。通常，范围在 [25,34] 之间的值是可以接受的结果，而34+ 被认为非常好。

* SSIM（结构相似性指数）是一个在 [0,1] 范围内的度量，其中1是完美匹配。最终指数由亮度、对比度和结构三个组件计算得出。如果你感兴趣，[这篇论文](https://arxiv.org/pdf/2006.13846.pdf)分析了SSIM及其组件。

* Inception Score 在 [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf) 中提出。它是使用 Inception-v3 模型的特征计算的，数值越高越好。虽然数学上很有趣，但最近不再被广泛使用。

* CLIP Score，这一指标在 [CLIPScore: A Reference-free Evaluation Metric for Image Captioning](https://arxiv.org/pdf/2104.08718.pdf) 中提出，用于评估文本到图像模型的质量。它通过使用 CLIP 模型计算生成图像和文本提示之间的余弦相似性，范围为 [0, 100]，值越高越好。

如果你对 FID *非常好奇*，[The Role of ImageNet Classes in Fréchet Inception Distance](https://arxiv.org/pdf/2203.06026.pdf) 试图分析 FID 在图像中认为重要的特征，以及在ImageNet上预训练的特征如何影响 FID 分数。这是一篇非常有趣的读物。