# MobileViT v2 

The previously discussed Vision Transformer architectures are computationally intensive and hard to run on mobile devices. The previous state-of-the-art architecture used CNNs for mobile vision tasks. However, CNNs cannot learn global representations, and as a result they perform worse than their transformer counterparts.

The MobileViT architecture aims to solve the required problems for vision mobile tasks, such as low-latency and lightweight architecture, while providing the advantages of transformers and CNNs. The mobileViT Architecture was developed by Apple and builds MobileNet from Google's research team. The MobileViT architecture builds upon the previous MobileNet architecture by adding the MobileViT Block and separable self-attention. These two features allow for lightning-fast latency, reduction of parameters, computational complexity, and deployment of vision ML models on resource-constrained devices.

## MobileViT Architecture

The architecture of MobileViT presented in the paper "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer" by Sachin Mehta and Mohammad Rastegari is as follows:
![MobileViT Architecture](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/MobileViT-Architecture.png)

Some of this should look similar to the previous chapter. The MobileNet blocks, nxn convolutions, downsampling, global pooling, and the final linear layer.

As seen by the global pooling layer and the linear layer, the model shown here is for classification. However, the same blocks introduced in this paper can be used for a variety of vision applications.

## MobileViT Block

The MobileViT block combines CNN's local processing and global processing, as seen in transformers. It uses a combination of convolutions and a transformer layer, allowing it to capture spatially local information and global dependencies in the data. 

A diagram of the MobileViT Block is shown below:
![MobileViT Block](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/MobileViT-MobileViTBlock.png)

Okay, that's a lot to take in. Let's break that down.

- The block takes in an image with multiple channels. Let's say for an RGB image 3 channels, so the block takes in a three channeled image. 
- It then performs a N by N convolution on the channels appending them to the existing channels.
- The block then creates a linear combination of these channels and adds them to the existing stack of channels.
- For each channel these images are unfolded into flattened patches.
- Then these flattened patches are passed through a transformer to project them into new patches. 
- These patches are then folded back together to create an image with d dimensions. 
- Afterwards a pointwise convolution is overlayed on the stitched image. 
- And then the stitched image is then recombined with the original RGB images.

This approach allows for a receptive field of H x W (the entire input size) while modeling non-local dependencies and local dependencies through retaining patch locational information. This can be seen in the unfolding and refolding of the patches.

<Tip>
A receptive field is the size of a region in an input space that affects the features of a particular layer.
</Tip>

This compound approach allows MobileViT to have fewer parameters than traditional CNNs and even better accuracy!
![MobileViT CNNPreformance](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/MobileViT-CNNPreformance.png)  

The main efficiency bottleneck in the original MobileViT architecture is the multi-head self-attention in Transformers, which requires O(k^2) time complexity concerning the input tokens.

Multi-head self-attention also requires costly operations like batch-wise matrix multiplications, which can impact latency on resource-constrained devices.

These same authors wrote another paper on exactly how to make attention operate faster. They've called it separable self-attention.

# Separable Self-attention

In traditional multihead attention, the big O concerning input tokens is quadratic (O(k^2)). Separable self-attention introduced in this paper has a complexity of O(k) concerning input tokens.

In addition, the attention method does not use any batch-wise matrix multiplications, which helps reduce latency on resource-constrained devices like mobile phones.

This is a massive improvement!

<Tip>
There have been many other forms of Attention such that complexity has ranged from O(k), O(k*sqrt(k)), O(k*log(k)).

Separable self-attention was not the first paper to have O(k) complexity. In Linformer, O(k) complexity for Attention was also achieved in [Linformer](https://arxiv.org/abs/2006.04768) before separable self-attention.

However, it still used costly operations like batch-wise matrix multiplications.
</Tip>

A comparison between the attention mechanisms in Transformer, Linformer, and MobileViT is shown below:
![Attention Comparison](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/MobileViT-Attention.png)  

The image above gives a comparison of each of the individual types of attention between the Transformer, Linformer, and MobileViT v2 architectures.

For example, in both the transformer and Linformer architectures, the attention computations perform two batch-wise matrix multiplications.  

However, in the case of separable self-attention, these two batch-wise multiplications are replaced by two separate linear computations. This allows for further boosted inference speed. 

# Conclusion

MobileViT blocks retain spatially local information while developing global representations, combining the strengths of Transformers and CNNs. They provide a receptive field that encompasses the entire image.

The introduction of separable self-attention into this existing architecture even further boosted both accuracy and inference speed.
![Inference Tests](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/MobileViT-Inference.png)

Tests performed with different architectures on the iPhone 12s exhibited a large jump in performance with the introduction of separable attention, as shown above!

Overall, the MobileViT Architecture is an extraordinarily powerful architecture for resource-limited vision tasks that provides fast inference and high accuracy.

# Transformers Library

If you want to try out MobileViTv2 locally, you can use it from HuggingFace's `transformers` library, here's how:

```bash
pip install transformers
```
Below is a short snippet on how to use MobileViT model to classify an image.

```python
from transformers import AutoImageProcessor, MobileViTV2ForImageClassification
from datasets import load_dataset
from PIL import Image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained(
    "apple/mobilevitv2-1.0-imagenet1k-256"
)
model = MobileViTV2ForImageClassification.from_pretrained(
    "apple/mobilevitv2-1.0-imagenet1k-256"
)

inputs = image_processor(image, return_tensors="pt")

logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

# Inference API

For an even lighter computer vision setup, you can use the Hugging Face Inference API with MobileViTv2.
Inference API is an API to interact with many models available on Hugging Face Hub.
We can query Inference API like following through Python.

```py
import json
import requests

headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = (
    "https://api-inference.huggingface.co/models/apple/mobilevitv2-1.0-imagenet1k-256"
)


def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


data = query("cats.jpg")
```
We can do the same with javascript like following.
```js
import fetch from "node-fetch";
import fs from "fs";
async function query(filename) {
    const data = fs.readFileSync(filename);
    const response = await fetch(
        "https://api-inference.huggingface.co/models/apple/mobilevitv2-1.0-imagenet1k-256",
        {
            headers: { Authorization: `Bearer ${API_TOKEN}` },
            method: "POST",
            body: data,
        }
    );
    const result = await response.json();
    return result;
}
query("cats.jpg").then((response) => {
    console.log(JSON.stringify(response));
});
```
Finally, we can query inference API through curl.
```bash
curl https://api-inference.huggingface.co/models/apple/mobilevitv2-1.0-imagenet1k-256 \
        -X POST \
        --data-binary '@cats.jpg' \
        -H "Authorization: Bearer ${HF_API_TOKEN}"
```