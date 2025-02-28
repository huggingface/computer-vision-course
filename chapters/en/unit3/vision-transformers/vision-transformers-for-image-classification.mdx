# Transfer Learning and Fine-tuning Vision Transformers for Image Classification

## Introduction

As the Transformers architecture scaled well in Natural Language Processing, the same architecture was applied to images by creating small patches of the image and treating them as tokens. The result was a Vision Transformer (ViT). Before we get started with transfer learning / fine-tuning concepts, let's compare Convolutional Neural Networks (CNNs) with Vision Transformers.

## Vision Transformer (VT) a Summary

To summarize, in Vision transformer, images are reorganized as 2D grids of patches. The models are trained on those patches.

The main idea can be found at the picture below: 
![Vision Transformer](https://huggingface.co/datasets/hf-vision/course-assets/blob/main/Screenshot%20from%202024-12-27%2014-25-49.png)

But there is a catch! The Convolutional Neural Networks (CNN)  are designed with an assumption missing in the VT. This assumption is based on how we perceive the objects in the images as humans. It is described in the following section.

## What are the differences between CNNs and Vision Transformers? 

### Inductive Bias

Inductive bias is a term used in machine learning to describe the set of assumptions that a learning algorithm uses to make predictions. In simpler terms, inductive bias is like a shortcut that helps a machine learning model make educated guesses based on the information it has seen so far.

Here's a couple of inductive biases we observe in CNNs:

- Translational Equivariance: an object can appear anywhere in the image, and CNNs can detect its features.
- Locality: pixels in an image interact mainly with its surrounding pixels to form features.

CNN models are very good at these two biases. ViT do not have this assumption. That is why for a dataset size up to a certain threshold actually CNNs are better than ViT. But ViT has another power!
The transformer architecture being (mostly) different types of linear functions allows ViT to become highly scalable. And that in turn allows ViT to overcome the problem of not having the above two
inductive biases with massive ammount of data!


### But how can everyone get access to massive datasets?

It's not feasible for everyone to train a Vision Transformer on millions of images to get good performance. Instead, one can use openly available model weights from places such as the [Hugging Face Hub](https://huggingface.co/models?sort=trending).

What do you do with the pre-trained model? You can apply transfer learning and fine-tune it!

## Transfer Learning & Fine-Tuning for Image Classification

The idea of transfer learning is that we can leverage the features learned by the Vision Transformers trained on a very large dataset and apply these features to our dataset. This can lead to significant improvements in model performance, especially when our dataset has limited data available for training.

Since we are taking advantage of the learned features, we do not need to update the entire model either. By freezing most of the weights, we can train only certain layers to get excellent performance with less training time and low GPU consumption.

### Multi-class Image Classification

You can go through the transfer learning tutorial using Vision Transformers for image classification in this notebook:

<a
  target="_blank"
  href="https://colab.research.google.com/github/johko/computer-vision-course/blob/main/notebooks/Unit%203%20-%20Vision%20Transformers/transfer-learning-image-classification.ipynb"
>
  <img
    src="https://colab.research.google.com/assets/colab-badge.svg"
    alt="Open In Colab"
  />
</a>

This is what we'll be building: an image classifier to tell apart dog and cat breeds:

<iframe
  src="https://shreydan-oxford-iiit-pets-classifier.hf.space"
  frameborder="0"
  width="850"
  height="450"
></iframe>

---

It might be that the domain of your dataset is not very similar to the pre-trained model's dataset. Yet, instead of training a Vision Transformer from scratch, we can choose to update the weights of the entire pre-trained model albeit with a lower learning rate, which will "fine-tune" the model to perform well with our data.

<Tip>
  However, in most scenarios, applying transfer learning is ample in the case of
  Vision Transformers.
</Tip>

### Multi-label Image Classification

The tutorial above teaches multi-class image classification, where each image only has 1 class assigned to it. What about scenarios where each image has multiple labels in a multi-class dataset?

This notebook will walk you through a fine-tuning tutorial using Vision Transformer for multi-label image classification:

<a
  target="_blank"
  href="https://colab.research.google.com/github/johko/computer-vision-course/blob/main/notebooks/Unit%203%20-%20Vision%20Transformers/fine-tuning-multilabel-image-classification.ipynb"
>
  <img
    src="https://colab.research.google.com/assets/colab-badge.svg"
    alt="Open In Colab"
  />
</a>

We'll also be learning how to use [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index) to write our custom training loops.
This is what you can expect to see as the outcome of the multi-label classification tutorial:

<iframe
  src="https://shreydan-pascal-multilabel-classifier.hf.space"
  frameborder="0"
  width="850"
  height="450"
></iframe>

---

### Additional Resources

- Original Vision Transformers Paper: _An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale [Paper](https://huggingface.co/papers/2010.11929)_
- Swin Transformers Paper: _Swin Transformer: Hierarchical Vision Transformer using Shifted Windows [Paper](https://huggingface.co/papers/2103.14030)_
- A systematic empirical study in order to better understand the interplay between the amount of training data, regularization, augmentation, model size and compute budget for Vision Transformers: _How to train your Vision Transformers? Data, Augmentation, and Regularization in Vision Transformers [Paper](https://huggingface.co/papers/2106.10270)_
