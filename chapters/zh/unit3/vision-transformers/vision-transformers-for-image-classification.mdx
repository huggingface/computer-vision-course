# 迁移学习与微调视觉Transformer用于图像分类

## 简介

随着Transformer架构在自然语言处理领域的成功扩展，同样的架构被应用到图像处理中，将图像划分成小块并将其视为“tokens”。最终产生了视觉Transformer（ViT）。在深入探讨迁移学习/微调概念之前，让我们先比较一下卷积神经网络（CNN）和视觉Transformer。

### CNN vs 视觉Transformer：归纳偏置

归纳偏置是机器学习中的一个术语，描述了学习算法用来进行预测的一组假设。简单来说，归纳偏置类似于一种捷径，帮助机器学习模型基于已经看到的信息做出合理的猜测。

以下是CNN中观察到的几种归纳偏置：

- 平移等变性：物体可以出现在图像的任意位置，CNN可以检测到它的特征。
- 局部性：图像中的像素主要与周围像素交互，以形成特征。

这些在视觉Transformer中并不存在。那么，它们是如何表现出色的呢？这是因为它们具有极强的可扩展性，并且在大量图像上进行训练，因此弥补了对这些归纳偏置的需求。

### 使用预训练的视觉Transformer

并非每个人都有能力在数百万张图像上训练一个视觉Transformer以获得良好的性能。相反，可以从诸如[Hugging Face Hub](https://huggingface.co/models?sort=trending)等开放平台中使用可用的模型。

使用预训练模型时该怎么做呢？可以应用迁移学习并对其进行微调！

## 用于图像分类的迁移学习与微调

迁移学习的思想是，我们可以利用在非常大数据集上训练的视觉Transformer所学习到的特征，将这些特征应用到我们的数据集中。这可以显著提升模型性能，特别是在我们的数据集训练数据有限的情况下。

由于我们利用了已学习的特征，通常不需要更新整个模型。通过冻结大部分权重，我们可以只训练某些层，从而以更少的训练时间和更低的GPU消耗获得出色的性能。

### 多类别图像分类

您可以在此笔记本中查看使用视觉Transformer进行图像分类的迁移学习教程：

<a
  target="_blank"
  href="https://colab.research.google.com/github/johko/computer-vision-course/blob/main/notebooks/Unit%203%20-%20Vision%20Transformers/transfer-learning-image-classification.ipynb"
>
  <img
    src="https://colab.research.google.com/assets/colab-badge.svg"
    alt="Open In Colab"
  />
</a>

我们将构建的内容是：一个可以区分狗和猫品种的图像分类器：

<iframe
  src="https://shreydan-oxford-iiit-pets-classifier.hf.space"
  frameborder="0"
  width="850"
  height="450"
></iframe>

---

可能您的数据集领域与预训练模型的数据集并不完全相似。然而，与其从头开始训练一个视觉Transformer，不如选择以较低的学习率更新整个预训练模型的权重，从而“微调”模型，以便在我们的数据上表现良好。

<Tip>
  不过，在大多数情况下，应用迁移学习对于视觉Transformer来说已经足够。
</Tip>

### 多标签图像分类

上述教程讲解了多类别图像分类，其中每个图像仅被分配一个类别。在每个图像在多类别数据集中有多个标签的情况下呢？

此笔记本将引导您完成使用视觉Transformer进行多标签图像分类的微调教程：

<a
  target="_blank"
  href="https://colab.research.google.com/github/johko/computer-vision-course/blob/main/notebooks/Unit%203%20-%20Vision%20Transformers/fine-tuning-multilabel-image-classification.ipynb"
>
  <img
    src="https://colab.research.google.com/assets/colab-badge.svg"
    alt="Open In Colab"
  />
</a>

我们还将学习如何使用[Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index)来编写自定义的训练循环。以下是多标签分类教程的预期结果：

<iframe
  src="https://shreydan-pascal-multilabel-classifier.hf.space"
  frameborder="0"
  width="850"
  height="450"
></iframe>

---

### 额外资源

- 原始视觉Transformer论文：_An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale [Paper](https://huggingface.co/papers/2010.11929)_
- Swin Transformer论文：_Swin Transformer: Hierarchical Vision Transformer using Shifted Windows [Paper](https://huggingface.co/papers/2103.14030)_
- 系统的实证研究，以更好地理解训练数据量、正则化、数据增强、模型大小和计算预算在视觉Transformer中的相互作用：_How to train your Vision Transformers? Data, Augmentation, and Regularization in Vision Transformers [Paper](https://huggingface.co/papers/2106.10270)_