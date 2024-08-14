# Convolutional Vision Transformer (CvT)

In this section, we will be doing a deep dive into Convolutional Vision Transformer (CvT) which is a variant of Vision Transformer (ViT)[[1]](#vision-transformer) and extensively used for the Image Classification task in Computer Vision.

## Recap

Before going into CvT, let's have a small recap on ViT architecture covered in the previous sections to better appreciate the CvT architecture. ViT decomposes each image into a sequence of tokens (i.e. non-overlapping patches) with a fixed length and then applies multiple standard Transformer layers, consisting of Multi-head Self Attention and Position-wise Feed-forward module (FFN) to model global relations for classification.

## Overview

Convolutional Vision Transformer (CvT) model was proposed in CvT: Introducing Convolutions to Vision Transformers [[2]](#cvt) by Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan and Lei Zhang. CvT employs all the benefits of CNNs: _local receptive fields_, _shared weights_, and _spatial subsampling_ along with _shift_, _scale_, _distortion invariance_ while keeping merits of Transformers: _dynamic attention_, _global context fusion_, and _better generalization_. CvT achieves superior performance while maintaining computational efficiency compared to ViT. Furthermore, due to built-in local context structure introduced by convolutions, CvT no longer requires a position embedding, giving it a potential advantage for adaption to a wide range of vision tasks requiring variable input resolution.

## Architecture

![CvT Architecture](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/cvt_architecture.png)
_(a) Overall architecture, showing the hierarchical multi-stage structure facilitated by the Convolutional Token Embedding layer. (b) Details of the Convolutional Transformer Block, which contains the convolutional projection as the first layer. [[2]](#cvt)_

The above image of CvT architecture illustrates the main steps of 3-stage pipeline. At its core, CvT blends two convolution-based operations into the Vision Transformer architecture:

- **Convolutional Token Embedding**: Imagine splitting the input image into overlapping patches, reshaping them into tokens, and then feeding them to a convolution layer. This reduces the number of tokens (like pixels in a downsampled image) while boosting their feature richness, similar to traditional CNNs. Unlike other Transformers, we skip adding pre-defined position information to the tokens, relying solely on convolutional operations to capture spatial relationships.

![Projection Layer](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/cvt_conv_proj.png)
_(a) Linear Projection in ViT. (b) Convolutional Projection. (c) Squeezed Convolutional Projection (Default in CvT). [[2]](#cvt)_

- **Convolutional Transformer Blocks**: Each stage in CvT contains a stack of these blocks. Here, instead of the usual linear projections in ViT, we use depth-wise separable convolutions (Convolutional Projection) to process the "query," "key," and "value" components of the self-attention module as shown in the above image. This maintains the benefits of Transformers while improving efficiency. Note that the "classification token" (used for final prediction) is only added in the last stage. Finally, a standard fully-connected layer analyzes the final classification token to predict the image class.

### Comparision of CvT Architecture with other Vision Transformers

The below table shows the key differences in terms of the necessity of positional encodings, type of token embedding, type of projection, and Transformer structure in the backbone, between the above representative concurrent works and CvT.

| Model                                            | Needs Position Encoding (PE) | Token Embedding                 | Projection for Attention | Hierarchical Transformers |
| ------------------------------------------------ | ---------------------------- | ------------------------------- | ------------------------ | ------------------------- |
| ViT[[1]](#vision-transformer), DeiT [[3]](#deit) | Yes                          | Non-overlapping                 | Linear                   | No                        |
| CPVT[[4]](#cpvt)                                 | No (w/PE Generator)          | Non-Overlapping                 | Linear                   | No                        |
| TNT[[5]](#tnt)                                   | Yes                          | Non-overlapping (Patch + Pixel) | Linear                   | No                        |
| T2T[[6]](#t2t)                                   | Yes                          | Overlapping (Concatenate)       | Linear                   | Partial (Tokenization)    |
| PVT[[7]](#pvt)                                   | Yes                          | Non-overlapping                 | Spatial Reduction        | Yes                       |
| _CvT_[[2]](#cvt)                                 | _No_                         | _Overlapping (Convolution)_     | _Convolution_            | _Yes_                     |

### Main Highlights

The four main highlights of CvT that helped achieve superior performance and computational efficiency are the following:

- **Hierarchy of Transformers** containing a new **Convolutional token embedding**.
- Convolutional Transformer block leveraging a **Convolutional Projection**.
- **Removal of Positional Encoding** due to built-in local context structure introduced by convolutions.
- Fewer **Parameters** and lower **FLOPs** (Floating Point Operations per second) compared to other vision transformer architectures.

## PyTorch Implementation

Time to get hands-on! Let's explore how to code each major blocks of the CvT architecture in PyTorch shown in the official implementation [[8]](#cvt-imp).

1. Importing required libraries

```python
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
```

2. Implementation of **Convolutional Projection**

```python
def _build_projection(self, dim_in, dim_out, kernel_size, padding, stride, method):
    if method == "dw_bn":
        proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            dim_in,
                            dim_in,
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                            bias=False,
                            groups=dim_in,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(dim_in)),
                    ("rearrage", Rearrange("b c h w -> b (h w) c")),
                ]
            )
        )
    elif method == "avg":
        proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "avg",
                        nn.AvgPool2d(
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                            ceil_mode=True,
                        ),
                    ),
                    ("rearrage", Rearrange("b c h w -> b (h w) c")),
                ]
            )
        )
    elif method == "linear":
        proj = None
    else:
        raise ValueError("Unknown method ({})".format(method))

    return proj
```

The method takes several parameters related to a convolutional layer (such as input and output dimensions, kernel size, padding, stride, and method) and returns a projection block based on the specified method.

- If the method is `dw_bn` (depthwise separable with batch normalization), it creates a Sequential block consisting of a depthwise separable convolutional layer followed by batch normalization and rearranges the dimensions.

- If the method is `avg` (average pooling), it creates a Sequential block with an average pooling layer followed by rearranging the dimensions.

- If the method is `linear`, it returns None, indicating that no projection is applied.

The rearrangement of dimensions is performed using the `Rearrange` operation, which reshapes the input tensor. The resulting projection block is then returned.

3. Implementation of **Convolutional Token Embedding**

```python
class ConvEmbed(nn.Module):
    def __init__(
        self, patch_size=7, in_chans=3, embed_dim=64, stride=4, padding=2, norm_layer=None
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x
```

This code defines a ConvEmbed module that performs patch-wise embedding on an input image.

- The `__init__` method initializes the module with parameters such as `patch_size` (size of the image patches), `in_chans` (number of input channels), `embed_dim` (dimensionality of the embedded patches), `stride` (stride for the convolution operation), `padding` (padding for the convolution operation), and `norm_layer` (a normalization layer, which is optional).

- In the constructor, a 2D convolutional layer (`nn.Conv2d`) is created with specified parameters, including the patch size, input channels, embedding dimension, stride, and padding. This convolutional layer is assigned to `self.proj`.

- If a normalization layer is provided, an instance of the normalization layer is created with embed_dim channels, and it is assigned to `self.norm`.

- The forward method takes an input tensor x and applies the convolution operation using `self.proj`. The output is reshaped using the rearrange function to flatten the spatial dimensions. If a normalization layer is present, it is applied to the flattened representation. Finally, the tensor is reshaped back to the original spatial dimensions and returned.

In summary, this module is designed for patch-wise embedding of images, where each patch is processed independently through a convolutional layer, and optional normalization is applied to the embedded features.

4. Implementation of **Vision Transformer** Block

```python
class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        patch_size=16,
        patch_stride=16,
        patch_padding=0,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init="trunc_norm",
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )

        with_cls_token = kwargs["with_cls_token"]
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)

        if init == "xavier":
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

        def forward(self, x):
            x = self.patch_embed(x)
            B, C, H, W = x.size()

            x = rearrange(x, "b c h w -> b (h w) c")

            cls_tokens = None
            if self.cls_token is not None:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)

            x = self.pos_drop(x)

            for i, blk in enumerate(self.blocks):
                x = blk(x, H, W)

            if self.cls_token is not None:
                cls_tokens, x = torch.split(x, [1, H * W], 1)
            x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

            return x, cls_tokens
```

This code defines a Vision Transformer module. Here's a brief overview of the code:

- **Initialization:** The `VisionTransformer` class is initialized with various parameters that define the model architecture, such as patch size, embedding dimensions, number of layers, number of attention heads, dropout rates, etc.

- **Patch Embedding:** The model includes a patch embedding layer (`patch_embed), which processes the input image by dividing it into non-overlapping patches and embedding them using Convolutions.

- **Transformer Blocks:** The model consists of a stack of transformer blocks (`Block`). The number of blocks is determined by the depth parameter. Each block contains multi-head self-attention mechanisms and a feedforward neural network.

- **Classification Token:** Optionally, the model can include a learnable classification token (`cls_token`) appended to the input sequence. This token is used for classification tasks.

- **Stochastic Depth:** Stochastic depth is applied to the transformer blocks, where a random subset of blocks is skipped during training to improve regularization. This is controlled by the `drop_path_rate` parameter.

- **Initialization of Weights:** The model weights are initialized using either truncated normal distribution (`trunc_norm`) or Xavier initialization (`xavier`).

- **Forward Method:** The forward method processes the input through the patch embedding, rearranges the dimensions, adds the classification token if present, applies dropout, and then passes the data through the stack of transformer blocks. Finally, the output is rearranged back to the original shape, and the classification token (if present) is separated from the rest of the sequence before returning the output.

5. Implementation of Convolutional Vision Transformer Block (**Hierarchy of Transformers**)

```python
class ConvolutionalVisionTransformer(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init="trunc_norm",
        spec=None,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.num_stages = spec["NUM_STAGES"]
        for i in range(self.num_stages):
            kwargs = {
                "patch_size": spec["PATCH_SIZE"][i],
                "patch_stride": spec["PATCH_STRIDE"][i],
                "patch_padding": spec["PATCH_PADDING"][i],
                "embed_dim": spec["DIM_EMBED"][i],
                "depth": spec["DEPTH"][i],
                "num_heads": spec["NUM_HEADS"][i],
                "mlp_ratio": spec["MLP_RATIO"][i],
                "qkv_bias": spec["QKV_BIAS"][i],
                "drop_rate": spec["DROP_RATE"][i],
                "attn_drop_rate": spec["ATTN_DROP_RATE"][i],
                "drop_path_rate": spec["DROP_PATH_RATE"][i],
                "with_cls_token": spec["CLS_TOKEN"][i],
                "method": spec["QKV_PROJ_METHOD"][i],
                "kernel_size": spec["KERNEL_QKV"][i],
                "padding_q": spec["PADDING_Q"][i],
                "padding_kv": spec["PADDING_KV"][i],
                "stride_kv": spec["STRIDE_KV"][i],
                "stride_q": spec["STRIDE_Q"][i],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs,
            )
            setattr(self, f"stage{i}", stage)

            in_chans = spec["DIM_EMBED"][i]

        dim_embed = spec["DIM_EMBED"][-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = spec["CLS_TOKEN"][-1]

        # Classifier head
        self.head = (
            nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        )
        trunc_normal_(self.head.weight, std=0.02)

    def forward_features(self, x):
        for i in range(self.num_stages):
            x, cls_tokens = getattr(self, f"stage{i}")(x)

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = torch.squeeze(x)
        else:
            x = rearrange(x, "b c h w -> b (h w) c")
            x = self.norm(x)
            x = torch.mean(x, dim=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x
```

This code defines a PyTorch module called `ConvolutionalVisionTransformer`.

- The model consists of multiple stages, each represented by an instance of the `VisionTransformer` class.
- Each stage has different configurations such as patch size, stride, depth, number of heads, etc., specified in the spec dictionary.
- The `forward_features` method processes the input x through all the stages, and it aggregates the final representation.
- The class has a classifier head that performs a linear transformation to produce the final output.
- The `forward` method calls `forward_features` and then passes the result through the classifier head.
- The vision transformer stages are sequentially named as stage0, stage1, etc., and each stage is an instance of the `VisionTransformer` class forming a hierarchy of transformers.

Congratulations! Now you know how to implement CvT architecture in PyTorch. You can view complete code of the CvT architecture [here](https://github.com/microsoft/CvT/blob/main/lib/models/cls_cvt.py).

## Try it out

If you're looking to use CvT without getting into the complex details of its PyTorch implementation, you can easily do so by leveraging the Hugging Face `transformers` library. Here's how:

```bash
pip install transformers
```

You can find the documentation for CvT model [here](https://huggingface.co/docs/transformers/model_doc/cvt#overview).

### Usage

Here is how to use CvT model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes:

```python
from transformers import AutoFeatureExtractor, CvtForImageClassification
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/cvt-13")
model = CvtForImageClassification.from_pretrained("microsoft/cvt-13")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) <a id="vision-transformer"></a>
- [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808) <a id="cvt"></a>
- [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) <a id="deit"></a>
- [Conditional Positional Encodings for Vision Transformers](https://arxiv.org/abs/2102.10882) <a id="cpvt"></a>
- [Transformer in Transformer](https://arxiv.org/abs/2103.00112v3)<a id="tnt"></a>
- [Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://arxiv.org/abs/2101.11986) <a id="t2t"></a>
- [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122) <a id="pvt"></a>
- [Implementation of CvT](https://github.com/microsoft/CvT/tree/main) <a id="cvt-imp"></a>
