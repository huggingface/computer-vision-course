# 稳定扩散简介

本章介绍了稳定扩散的构建模块，这是一种生成式人工智能（generative AI）模型，可以根据文本和图像提示生成独特的写实图像。该模型最初于2022年推出，由[Stability AI](https://stability.ai/)、[RunwayML](https://runwayml.com/)和LMU慕尼黑的CompVis小组合作开发，并依据[论文](https://arxiv.org/pdf/2112.10752.pdf)提出。

你将从本章中学到什么？
- 稳定扩散的基本组成部分
- 如何使用`text-to-image`、`image2image`和修复管道

## 稳定扩散需要什么才能工作？

为了让本节更具趣味性，我们将尝试通过一些问题来理解稳定扩散过程的基本组成部分。我们会简要讨论每个组成部分，因为它们已在我们的扩散器课程中详细介绍。你也可以访问我们之前的章节，深入了解GANS和扩散模型的详细内容。

- 稳定扩散采用了哪些策略来学习新信息？
    - 它使用了扩散模型的前向和反向过程。在前向过程中，我们向图像中加入高斯噪声，直到剩下的仅是随机噪声。通常，我们无法识别出图像的最终噪声版本。
    - 在反向过程中，我们有一个经过训练的神经网络来从纯噪声逐步去噪，最终生成一个实际图像。

这两个过程都是在有限的步数`T`内完成的（根据DDPM论文，T=1000）。你从时间$t_0$开始这个过程，通过从数据分布中采样真实图像，前向过程在每个时间步t从高斯分布中采样噪声并将其添加到上一步的图像中。如需更多数学直观的理解，请阅读[Hugging Face Blog](https://huggingface.co/blog/annotated-diffusion)上的扩散模型内容。

- 由于图像可能非常大，如何压缩它们？

当你有大型图像时，它们需要更多的计算资源来处理，特别是在一个名为自注意力的操作中。图像越大，所需的计算量越多，并且这些计算量会随着图像大小快速增加（数学上称为“二次增长”）。
例如，如果图像是128像素宽和高，其像素数量是64像素宽和高的四倍。由于自注意力的工作原理，处理这个更大的图像不仅需要四倍的内存和计算资源，实际上需要十六倍的资源（因为4乘4等于16）。这使得处理非常高分辨率的图像变得具有挑战性，因为它们需要大量的资源。
潜变量扩散模型通过使用变分自编码器（VAE）将图像缩小到更易处理的大小，以应对处理大型图像的高计算需求。该思想是许多图像中存在重复或不必要的信息。经过大量数据训练的VAE可以将图像压缩为一个更小的、简洁的形式，而这一缩小的版本仍然保留了原始图像的基本特征。

- 既然我们在使用提示，如何将文本与图像融合？

我们知道，在推理时，我们可以输入我们想要看到的图像描述，以及一些作为起点的纯噪声，模型会尽力将随机输入“去噪”成与标题匹配的内容。
稳定扩散利用了一个基于[CLIP](https://huggingface.co/learn/computer-vision-course/unit4/multimodal-models/clip-and-relatives/clip)的预训练Transformer模型。CLIP的文本编码器被设计用于处理图像标题，将其转换为可用于比较图像和文本的形式，因此非常适合从图像描述中创建有用的表示。输入的提示首先被标记化（基于一个大型词汇表，每个单词或子词都分配一个特定的标记），然后通过CLIP文本编码器生成一个768维（在SD 1.X中）或1024维（SD 2.X中）的向量表示每个标记。为了保持一致性，提示始终填充/截断为77个标记，因此我们用作条件的最终表示是每个提示为77x1024的张量。

- 如何加入良好的归纳偏差？

因为我们尝试生成一些新的东西（例如，一个逼真的Pokemon），我们需要一种方法来超越我们之前见过的图像（例如，一个动漫风格的Pokemon）。这就是U-Net和自注意力的作用所在。给定图像的一个噪声版本，模型的任务是根据诸如图像的文本描述等附加线索预测去噪版本。那我们实际上如何将这种条件信息输入到U-Net中让其在进行预测时使用呢？答案是交叉注意力。在U-Net的不同位置上有交叉注意力层。
U-Net中的每个空间位置都可以“关注”文本条件中的不同标记，从提示中引入相关信息。

## 如何在Diffusers中使用`text-to-image`、`image-to-image`和修复模型

本节介绍了实用的使用案例以及如何使用[Diffusers](https://github.com/huggingface/diffusers)库来执行这些任务。
- `text-to-image`推理的步骤
想法是传递文本提示，将其转换为输出图像。

<iframe
	src="https://hysts-controlnet-v1-1.hf.space/"
	frameborder="0"
	width="850"
	height="450">
</iframe>

使用`diffusers`库，你可以通过2个步骤实现`text-to-image`功能。

首先，让我们安装`diffusers`库。
```bash
pip install diffusers
```

我们现在将初始化管道，传递我们的提示，并进行推理。
```python
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
generator = torch.Generator(device="cuda").manual_seed(31)
image = pipeline(
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    generator=generator,
).images[0]
```

- `image-to-image`推理的步骤
同样，我们可以初始化管道，但传递图像和文本提示。
```python
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipeline.enable_model_cpu_offload()
# 如果未安装xFormers或已安装PyTorch 2.0或更高版本，请删除以下行
pipeline.enable_xformers_memory_efficient_attention()

# 加载要传递给管道的图像：
init_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
)

# 传递一个提示和图像到管道生成图像：
prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
image = pipeline(prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

- 修复的步骤
对于修复管道，我们需要传递一个图像、文本提示和一个基于图像中对象的掩码，该掩码指示图像中要修复的内容。
在这个例子中，我们还传递了一个负提示，以进一步影响推理，表明我们想要避免的内容。
```python
# 加载管道
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()
# 如果未安装xFormers或已安装PyTorch 2.0或更高版本，请删除以下行
pipeline.enable_xformers_memory_efficient_attention()

# 加载基础和掩码图像：
init_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
)
mask_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"
)

# 创建一个用于修复图像的提示，并将其与基础图像和掩码图像一起传递到管道：
prompt = (
    "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
)
negative_prompt = "bad anatomy, deformed, ugly

, disfigured"
image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    mask_image=mask_image,
).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

### 延伸阅读
- [Diffusers文档](https://huggingface.co/docs/diffusers/using-diffusers/pipeline_overview)
- [Diffusers安装](https://huggingface.co/docs/diffusers/installation)