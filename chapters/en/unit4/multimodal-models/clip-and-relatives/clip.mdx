# Contrastive Language-Image Pre-training (CLIP)

## Introduction

CLIP is a neural network adept at grasping visual concepts through natural language supervision. It operates by concurrently training a text encoder and an image encoder, focusing on a pretraining task that involves matching captions with corresponding images. This architecture allows CLIP to adapt to a variety of visual classification benchmarks seamlessly. It does so by simply receiving the names of the visual categories to be recognized, demonstrating "zero-shot" learning capabilities akin to those observed in GPT-2 and GPT-3 models.

## Contrastive pre-training

Given a batch of image-text pairs, CLIP computes the dense cosine similarity matrix between all possible (image, text) candidates within this batch. The core idea is to maximize the similarity between the correct pairs (shown in blue in the figure below) and minimize the similarity for incorrect pairs (shown in grey in the image). To do it, they optimize a symmetric cross-entropy loss over these similarity scores. 

![CLIP contrastive pre-training](https://images.openai.com/blob/fbc4f633-9ad4-4dc2-bd94-0b6f1feee22f/overview-a.svg)
_Image taken from OpenAI_

Explaining in simple terms, we want to make the similarity between the image and its corresponding caption as high as we can, while the similarity between the image and the other captions should be small. We apply this logic to the caption too, so we want to maximize the similarity of the caption with its corresponding image, and minimize between all other images.

## Text Encoder and Image Encoder

CLIP's design features independent encoders for images and text, allowing flexibility in their choice. Users can switch the standard image encoder, like a Vision Transformer, for alternatives like ResNet, or opt for different text encoders, enhancing adaptability and experimentation. Of course, if you switch one of the encoders, you will need to train your model again, as your embedding distribution will be different.

## Use cases
CLIP, can be leveraged for a variety of applications. Here are some notable use cases:

- Zero-shot image classification;
- Similarity search;
- Diffusion models conditioning.

## Usage

For practical applications, one typically uses an image, and pre-defined classes as input. The provided Python example demonstrates how to use the transformers library for running CLIP. In this example, we want to zero-shot classify the image below between `dog` or `cat`.

![A photo of cats](http://images.cocodataset.org/val2017/000000039769.jpg)

```python
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=image,
    return_tensors="pt",
    padding=True,
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
```

After executing this code, we got the following probabilities:
- "a photo of a cat": 99.49%
- "a photo of a dog": 0.51%

## Limitations

Despite CLIP's proficiency in zero-shot classification, it is unlikely to outperform a specialized, fine-tuned model. Moreover, its generalization capabilities are somewhat limited, particularly in scenarios involving data or examples not encountered during training.
The paper also shows that CLIP's effectiveness and biases are impacted by the choice of categories, as demonstrated in tests using the Fairface dataset. Notable disparities were found in gender and racial classifications, with gender accuracy over 96% and racial accuracy around 93%.

## Conclusion

In conclusion, the CLIP model from OpenAI has revolutionized the multimodal field. What sets CLIP apart is its proficiency in zero-shot learning, allowing it to classify images into categories it wasn't explicitly trained on. This remarkable ability to generalize comes from its innovative training method, where it learns to match images with text captions.

## References

- [CLIP paper](https://arxiv.org/abs/2103.00020)
- [CLIP by Lilian Weng](https://lilianweng.github.io/posts/2021-05-31-contrastive/#clip)