# OneFormer: One Transformer to Rule Universal Image Segmentation

<img align="center" width="1050" height="303" src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/oneformer/teaser.svg">

## Introduction

OneFormer is a groundbreaking approach to image segmentation, a computer vision task that involves dividing an image into meaningful segments. Traditional methods used separate models and architectures for different segmentation tasks, like identifying objects (instance segmentation) or labeling regions (semantic segmentation). More recent attempts aimed to unify these tasks using shared architectures but still required separate training for each task.

Enter OneFormer, a universal image segmentation framework designed to overcome these challenges. It introduces a unique multi-task approach, allowing a single model to handle semantic, instance, and panoptic segmentation tasks without the need for separate training on each. The key innovation lies in a task-conditioned joint training strategy, where the model is guided by a task input, making it dynamic and adaptive to different tasks during both training and inference.

This breakthrough not only simplifies the training process but also outperforms existing models across various datasets. OneFormer achieves this by using panoptic annotations, unifying the ground truth information needed for all tasks. Additionally, the framework introduces query-text contrastive learning to better distinguish between tasks and improve overall performance.

## Background of OneFormer

![Oneformer Method](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/oneformer/oneformer.svg)
_Image taken from OneFormer paper_

To understand OneFormer's significance, let's look into the background of image segmentation. In image processing, segmentation involves dividing an image into different parts, which is crucial for tasks like recognizing objects and understanding the content of a scene. Traditionally, two main types of segmentation tasks were semantic segmentation, where pixels are labeled with categories like "road" or "sky," and instance segmentation, which identifies objects with well-defined boundaries.

Over time, researchers proposed panoptic segmentation as a way to unify semantic and instance segmentation tasks. However, even with these advancements, there were challenges. Existing models designed for panoptic segmentation still required separate training for each task, making them semi-universal at best.

This is where OneFormer comes in as a game-changer. It introduces a novel approach – a multi-task universal architecture. The idea is to train this framework only once, using a single universal architecture, a lone model, and just one dataset. The magic happens as OneFormer outperforms specialized frameworks across semantic, instance, and panoptic segmentation tasks. This breakthrough is not just about improving accuracy; it's about making image segmentation more universal and efficient. With OneFormer, the need for extensive resources and separate training for different tasks becomes a thing of the past.

## Core Concepts of OneFormer

![Task Conditioned Joint Training](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/oneformer/text_gen.svg)

Now, let's break down the key features of OneFormer that make it stand out:

### Task-Dynamic Mask

OneFormer uses a clever trick called "Task-Dynamic Mask" to better understand and tackle different types of image segmentation tasks. So, when the model encounters an image, it uses this "Task-Dynamic Mask" to decide whether to pay attention to the overall scene, identify specific objects with clear boundaries, or do a combination of both.

### Task-Conditioned Joint Training
One of the groundbreaking features of OneFormer is its task-conditioned joint training strategy. Instead of training separately for semantic, instance, and panoptic segmentation, OneFormer uniformly samples the task during training. This strategy enables the model to learn and generalize across different segmentation tasks simultaneously. By conditioning the architecture on the specific task through the task token, OneFormer unifies the training process, reducing the need for task-specific architectures, models, and datasets. This innovative approach significantly streamlines the training pipeline and resource requirements.

### Query-Text Contrastive Loss

Lastly, let's talk about "Query-Text Contrastive Loss." Think of it as a way for OneFormer to teach itself about the differences between tasks and classes. In the training process, the model compares the features it extracts from the image (queries) with the corresponding text descriptions (like "a photo with a car"). This helps the model understand the unique characteristics of each task and reduces confusion between different classes. OneFormer's "Task-Dynamic Mask" allows it to be versatile like a multitasking assistant, and the "Query-Text Contrastive Loss" helps it learn the specifics of each task by comparing visual features with textual descriptions.

By combining these core concepts, OneFormer becomes a smart and efficient tool for image segmentation, making the process more universal and accessible. 

## Conclusion

![result comperison](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/oneformer/plots.svg)
_Image taken from OneFormer paper_

In conclusion, the OneFormer framework represents a groundbreaking approach in the field of image segmentation, aiming to simplify and unify the task across various domains. Unlike traditional methods that rely on specialized architectures for each segmentation task, OneFormer introduces a novel multi-task universal architecture that requires only a single model, trained once on a universal dataset, to outperform existing frameworks. Additionally, the incorporation of query-text contrastive loss during training enhances the model's ability to learn inter-task and inter-class differences. OneFormer utilizes transformer-based architectures, inspired by recent successes in computer vision, and introduces task-guided queries to improve task sensitivity. The results are impressive, as OneFormer surpasses state-of-the-art models across semantic, instance, and panoptic segmentation tasks on benchmark datasets like ADE20k, Cityscapes, and COCO. The framework's performance is further enhanced with new ConvNeXt and DiNAT backbones.

In summary, OneFormer represents a significant step towards universal and accessible image segmentation. By introducing a single model capable of handling diverse segmentation tasks, the framework streamlines the segmentation process and reduces resource requirements.

## Example use of model

Let's see an example use of the model. Dinat backbone requires Natten library, which may take a while to install.
```bash
!pip install -q natten 
```
We can see an inference code below depending on different segmentation types.

```python
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt


def run_segmentation(image, task_type):
    """Performs image segmentation based on the given task type.

    Args:
        image (PIL.Image): The input image.
        task_type (str): The type of segmentation to perform ('semantic', 'instance', or 'panoptic').

    Returns:
        PIL.Image: The segmented image.

    Raises:
        ValueError: If the task type is invalid.
    """

    processor = OneFormerProcessor.from_pretrained(
        "shi-labs/oneformer_ade20k_dinat_large"
    )  # Load once here
    model = OneFormerForUniversalSegmentation.from_pretrained(
        "shi-labs/oneformer_ade20k_dinat_large"
    )

    if task_type == "semantic":
        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        outputs = model(**inputs)
        predicted_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]

    elif task_type == "instance":
        inputs = processor(images=image, task_inputs=["instance"], return_tensors="pt")
        outputs = model(**inputs)
        predicted_map = processor.post_process_instance_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]["segmentation"]

    elif task_type == "panoptic":
        inputs = processor(images=image, task_inputs=["panoptic"], return_tensors="pt")
        outputs = model(**inputs)
        predicted_map = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]["segmentation"]

    else:
        raise ValueError(
            "Invalid task type. Choose from 'semantic', 'instance', or 'panoptic'"
        )

    return predicted_map


def show_image_comparison(image, predicted_map, segmentation_title):
    """Displays the original image and the segmented image side-by-side.

    Args:
        image (PIL.Image): The original image.
        predicted_map (PIL.Image): The segmented image.
        segmentation_title (str): The title for the segmented image.
    """

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_map)
    plt.title(segmentation_title + " Segmentation")
    plt.axis("off")
    plt.show()


url = "https://huggingface.co/datasets/shi-labs/oneformer_demo/resolve/main/ade20k.jpeg"
response = requests.get(url, stream=True)
response.raise_for_status()  # Check for HTTP errors
image = Image.open(response.raw)

task_to_run = "semantic"
predicted_map = run_segmentation(image, task_to_run)
show_image_comparison(image, predicted_map, task_to_run)
```
![semantic segmentation](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/oneformer/oneformer_semantic.png)

## References

- [OneFormer Paper](https://arxiv.org/pdf/2211.06220.pdf)  
- [HuggingFace OneFormer model](https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large)
