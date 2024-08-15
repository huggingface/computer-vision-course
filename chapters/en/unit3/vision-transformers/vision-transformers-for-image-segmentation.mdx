# Transformer-based image segmentation

In this section, we'll explore how Vision Transformers compare to Convolutional Neural Networks (CNNs) in image segmentation and detail the architecture of a vision transformer-based segmentation model as an example.

<Tip warning={true}>
  This section assumes familiarity with image segmentation, Convolutional Neural
  Networks (CNNs), and the basics of Vision Transformers. If you're new to these
  concepts, we recommend exploring related materials in the course before
  proceeding.
</Tip>

## CNNs vs Transformers for Segmentation

Before the emergence of Vision Transformers, CNNs (Convolutional Neural Networks) have been the go-to choice for image segmentation. Models like [U-Net](https://arxiv.org/abs/1505.04597) and [Mask R-CNN](https://arxiv.org/abs/1703.06870) captured the details that are needed to distinguish different objects in an image, making them state-of-the-art for segmentation tasks.

Despite their excellent results over the past decade, CNN-based models have some limitations, which Transformers aim to solve:

- **Spatial limitations**: CNNs learn local patterns through small receptive fields. This local focus makes it hard for them to "link" features that are far apart but related within the image, affecting their ability to accurately segment complex scenes/objects. Unlike CNNs, ViTs are designed to capture global dependencies within an image, leveraging the attention mechanism. This means ViT-based models consider the entire image at once, allowing them to understand complex relationships between distant parts of an image. For segmentation, this global perspective can lead to a more accurate delineation of objects.
- **Task-Specific Components**: Methods like Mask R-CNN incorporate hand-designed components (e.g., non-maximum suppression, spatial anchors) to encode prior knowledge about segmentation tasks. These components add complexity and require manual tuning. In contrast, ViT-based segmentation methods simplify the segmentation process by eliminating the need for hand-designed components, making them more straightforward to optimize.
- **Segmentation Task Specialization**: CNN-based segmentation models approach semantic, instance, and panoptic segmentation tasks individually, leading to specialized architectures for each task and separate research efforts into each. Recent ViT-based models like [MaskFormer](https://arxiv.org/abs/2107.06278), [SegFormer](https://arxiv.org/abs/2105.15203) or [SAM](https://arxiv.org/abs/2304.02643) provide a unified approach to tackling semantic, instance, and panoptic segmentation tasks within a single framework.

## Spotlight on MaskFormer: Illustrating ViT for Image Segmentation

MaskFormer ([paper](https://arxiv.org/abs/2107.06278), [Hugging Face transformers documentation](https://huggingface.co/docs/transformers/en/model_doc/maskformer)), introduced in the paper "MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation" is a model that predicts segmentation masks for each class present in an image, unifying semantic and instance segmentation in one architecture.

### MaskFormer Architecture

The figure below shows the architecture diagram taken from the paper.

<img
  width="600"
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/maskformer_architecture.png"
/>

The architecture is composed of three components:

**Pixel-level Module**: Uses a backbone to extract image features and a pixel decoder to generate per-pixel embeddings.

**Transformer Module**: Employs a standard Transformer decoder to compute per-segment embeddings from image features and learnable positional embeddings (queries), encoding global information about each segment.

**Segmentation Module**: Generates class probability predictions and mask embeddings for each segment using a linear classifier and a Multi-Layer Perceptron (MLP), respectively. The mask embeddings are used in combination with per-pixel embeddings to predict binary masks for each segment.

The model is trained with a binary mask loss, the same one as [DETR](https://huggingface.co/learn/computer-vision-course/unit3/vision-transformers/detr), and a cross-entropy classification loss per predicted segment.

### Panoptic Segmentation Inference Example with Hugging Face Transformers

Panoptic segmentation is the task of labeling every pixel in an image with its category and identifying distinct objects within those categories, combining both semantic and instance segmentation.

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

As you can see below, the results include multiple instances of the same classes, each with distinct masks.

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

## Fine-tuning Vision Transformer-based Segmentation Models

With many pre-trained segmentation models available, transfer learning and finetuning are commonly used to adapt these models to specific use cases, especially since transformer-based segmentation models, like MaskFormer, are data-hungry and challenging to train from scratch.
These techniques leverage pre-trained representations to adapt these models to new data efficiently. Typically, for MaskFormer, the backbone, the pixel decoder, and the transformer decoder are kept frozen to leverage their learned general features, while the transformer module is finetuned to adapt its class prediction and mask generation capabilities to new segmentation tasks.

[This notebook](https://colab.research.google.com/github/johko/computer-vision-course/blob/main/notebooks/Unit%203%20-%20Vision%20Transformers/transfer-learning-segmentation.ipynb) will walk you through a transfer learning tutorial on image segmentation using MaskFormer.

## References

- [MaskFormer Hugging Face documentation](https://huggingface.co/docs/transformers/en/model_doc/maskformer)
- [Image Segmentation Hugging Face Task Guide](https://huggingface.co/docs/transformers/en/tasks/semantic_segmentation)
