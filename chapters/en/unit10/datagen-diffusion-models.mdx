# Synthetic Data Generation with Diffusion Models

Imagine trying to train a model for tumor segmentation. As it's hard to gather data for medical imaging, it'd be really difficult for the model to converge.
Ideally, we expect to have at least enough data to build a simple baseline, but what if you have just a few samples? Synthetic data generation methods try to solve this dilemma, and now we have many more options with the boom of generative models!

As you've seen in the previous sections, it is possible to use generative models such as DCGAN to generate synthetic images. In this section, we will focus on diffusion models using [diffusers](https://huggingface.co/docs/diffusers/index)!

## Recap of the Diffusion Models

Diffusion models are generative models that gained significant popularity in recent years, thanks to their capabilities to produce high-quality images. Nowadays, they're widely used in image, video, and text synthesis.

A diffusion model works by learning to denoise random Gaussian noise step-by-step. The training process requires adding Gaussian noise to the input samples and letting the model learn denoising. 

Diffusion models are generally conditioned to a kind of input besides the data distribution, such as text prompts, images or even audio. Additionally, it's also possible to [build an unconditional generator](https://huggingface.co/docs/diffusers/training/unconditional_training).

There are many underlying concepts behind the model's inner workings but the simplified version is similar to this:

Firstly, the model adds noise to the input and processes it.

![noising](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/noising.jpg?download=true)

Then the model learns to denoise the given data distribution.

![denoising](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/denoising.jpg?download=true)

We won't dive deep into the theory, but understanding how a diffusion model works will come really handy when we need to pick a technique to generate synthetic data for our use case.

## Text-To-Image Diffusion Model: Stable Diffusion

Essentially, the way Stable Diffusion (SD) works is the same as we mentioned above. It uses three main components that help it to produce high-quality images.

1. **The diffusion process:** The input is processed multiple times to generate useful information about the image. The "usefulness" is learned while training the model.

2. **Image encoder and decoder model:** Lets the model compress images from pixel space to a smaller dimensional space, abstracting meaningless information while increasing performance.

3. **Optional conditional encoder:** This component is used to condition to generation process on an input. This extra input can be text prompts, images, audio, and other representations. Originally, it was a text encoder.

So while the general workflow looks like this:
![general stable diffusion workflow](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/general-workflow.png?download=true)

Originally we used a text encoder to condition the model on our prompts:
![original stable diffusion workflow](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/stable-diffusion-workflow.png?download=true)

This short overview is just the tip of the iceberg! If you wish to dive deep into the theory behind stable diffusion (or diffusion models), you can out the [further reading section](#further-reading-about-stable-diffusion)!

[diffusers](https://huggingface.co/docs/diffusers/index) provide us ready to use pipelines for different tasks, such as:

| Task | Description | Pipeline |
|------|-------------|----------|
| Unconditional Image Generation | generate an image from Gaussian noise | [unconditional_image_generation](https://huggingface.co/docs/diffusers/using-diffusers/unconditional_image_generation) |
| Text-Guided Image Generation | generate an image given a text prompt |[conditional_image_generation](https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation) |
| Text-Guided Image-to-Image Translation | adapt an image guided by a text prompt | [img2img](https://huggingface.co/docs/diffusers/using-diffusers/img2img) |
| Text-Guided Image-Inpainting | fill the masked part of an image given the image, the mask and a text prompt | [inpaint](https://huggingface.co/docs/diffusers/using-diffusers/inpaint) |
| Text-Guided Depth-to-Image Translation | adapt parts of an image guided by a text prompt while preserving structure via depth estimation | [depth2img](https://huggingface.co/docs/diffusers/using-diffusers/depth2img) |

There's also a complete list of supported tasks that you can find from the [Diffusers Summary](https://huggingface.co/docs/diffusers/api/pipelines/overview#diffusers-summary) table.

This means we have many tools under our belt to generate synthetic data!

## Approaches to Synthetic Data Generation

There are generally three cases for needing synthetic data:

**Extending an existing dataset:**

- **There are not enough samples:** A nice example is a medical imaging dataset such as [DDSM](https://www.mammoimage.org/databases/) (Digital Database for Screening Mammography, ~2500 samples), a small amount of samples makes it harder to build a model for further analysis. It's also rather expensive to build such medical imaging datasets.

**Creating a Dataset from Scratch**:

- **There aren't any samples at all:** Let's assume that you want to build a weapon detection system on top of CCTV video streams. But there aren't any samples for the specific weapon you want to detect, you can use similar observations in different settings to apply style transfer to make them look like CCTV streams!

**Preserving Privacy**:

- Hospitals collect huge amounts of data on patients, surveillance cameras capture raw information about individuals' faces and activities, and all of these introduce a potential infringe on privacy. We can use diffusion models to generate privacy-preserving datasets to develop our solutions, without giving up on anyone's privacy rights.

There are different methods for us to utilize a text-to-image diffusion model to generate customized outputs. For example, by simply utilizing the pre-trained diffusion model (such as [Stable Diffusion XL](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)), you can try to construct a nice prompt to generate images. But the quality of the generated images may not be consistent and it might be really difficult to construct such a prompt for your specific use case.

You will usually be required to change some parts of the model to generate the personalized output you want, here are a few techniques that you can use for that:

**Training with [Textual Inversion](https://huggingface.co/docs/diffusers/main/en/training/text_inversion):**

![textual-inversion](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/textual_inversion.png?download=true)

Textual inversion is a technique that works by interfering with the text embeddings in the model's architecture. You can add a new token to the vocabulary and then fine-tune the embeddings using a few examples.

By providing samples corresponding to the new token, we try to optimize the embeddings to capture the characteristics of the object.

**Training a [LoRA (Low-Rank Adaptation)](https://huggingface.co/docs/diffusers/main/en/training/lora?installation=PyTorch) model:**

LoRA has targeted the problem of fine-tuning LLMs. It represents the weight updates with two smaller update matrices through low-rank decomposition, which has substantially less amount of parameters. Then these matrices can be trained to adapt the new data!

The base model's weights remain frozen in the whole process, so we just train the new update matrices. Then lastly, the adapted weights are combined with the original weights.

This means training LoRA is essentially much faster than full model fine-tuning! As an important note, it's also possible to combine LoRA with other techniques since it can be added on top of the model itself.

**Training with [DreamBooth](https://huggingface.co/docs/diffusers/main/en/training/dreambooth):**

![dreambooth](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/dreambooth.png?download=true)

DreamBooth is a technique to fine-tune the model to personalize the outputs. Given a few images of a subject, it lets you fine-tune a pre-trained text-to-image model. The main idea is to associate a unique identifier with that specific subject.

For training, we use the tokens in the vocabulary and build a dataset using preferably a rare-token identifier. Because if you choose a rather common identifier, the model would also have to learn to disentangle from their original meaning.

In the original paper, authors find the rare tokens in the vocabulary and then choose identifiers from those. This reduces the risk of an identifier having a strong prior. It is also stated that the best results were achieved by fine-tuning all the layers of the model.

**Training with [Custom Diffusion](https://huggingface.co/docs/diffusers/main/en/training/custom_diffusion):**

![custom diffusion](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/custom_diffusion.png?download=true)

Custom Diffusion is a really powerful technique to personalize the model. It requires just a few samples like the previously mentioned methods, but its power lies in being able to learn multiple concepts at the same time!

It works by training just a part of the diffusion process and the text encoder we mentioned above, meaning that there are fewer parameters to optimize. Thus, this method also enables fast fine-tuning!

## Real-world Examples of Using Diffusion Models for Dataset Generation

There are many unique cases where diffusion models are used to generate synthetic datasets!

Find them below.

- [Apple Images for Apple Detection in Orchards](https://arxiv.org/abs/2306.09762)
- [Automatically Labeled Polyp Images for Medical Image Segmentation](https://arxiv.org/abs/2310.16794)
- [3D Medical Image Generation with DDPMs](https://www.nature.com/articles/s41598-023-34341-2.pdf)
- [Generating Transmission Line Images with DDPM](https://ieeexplore.ieee.org/document/10281144)
- [Synthetic Aerial Dataset for Unmanned Aerial Vehicle Detection](https://ieeexplore.ieee.org/document/10195076)
- [Differentially Private Diffusion Models for Privacy-preserving Synthetic Image Generation](https://arxiv.org/pdf/2302.13861.pdf)

## Further reading about Stable Diffusion

- [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)
- [Diffusion Explainer: Stable Diffusion Explained with Visualization](https://poloclub.github.io/diffusion-explainer/)
- [Introduction to Diffusion Models](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)
- [FastAI, Practical Deep Learning for Coders - Lesson 9: Stable Diffusion](https://course.fast.ai/Lessons/lesson9.html)
- [Original paper](https://arxiv.org/abs/2112.10752)
