# Introduction to Stable Diffusion
This chapter introduces the building blocks of Stable Diffusion which is a generative artificial intelligence (generative AI) model that produces unique photorealistic images from text and image prompts. It originally launched in 2022 and was made possible thanks to a collaboration with
[Stability AI](https://stability.ai/), [RunwayML](https://runwayml.com/) and CompVis Group at LMU Munich following the [paper](https://arxiv.org/pdf/2112.10752.pdf).

What will you learn from this chapter?
- Fundamental components of Stable Diffusion
- How to use `text-to-image`, `image2image`, inpainting pipelines

## What Do We Need for Stable Diffusion to Work?
To make this section interesting we will try to answer some questions to understand the basic components of the Stable Diffusion process. 
We will briefly discuss each component as they are already covered in our Diffusers course. Also, you can visit our previous section, which talks about GANS and Diffusion models in details.

- What strategies does Stable Diffusion employ to learn new information?
    - It uses forward and reverse processes of diffusion models. In the forward process, we add Gaussian noise to an image until all that remains is the random noise. Usually we cannot identify the final noisy version of the image.
    - In the reserve process, we have a learned neural network trained to gradually denoise an image starting from pure noise, until you end up with an actual image.

Both of these processes happens for a finite number of steps `T`(as per DDPM paper T=1000). You begin the process at time \\(t_0\\) by sampling a real image from your data distribution, and the forward process samples some noise from a Gaussian distribution at each time step t, which is added to the image of the previous time step. To get more mathematical intuition, please read [Hugging Face Blog](https://huggingface.co/blog/annotated-diffusion) on Diffusion Models. 

- Since our images can be huge how can we compress it?

When you have large images, they require more computing power to process. This becomes very noticeable in a specific operation known as self-attention. The bigger the image, the more calculations are needed, and these calculations increase very quickly (in a way mathematicians call "quadratically") with the size of the image.
For example, if you have an image that's 128 pixels wide and tall, it has four times more pixels than an image that's only 64 pixels wide and tall. Because of how self-attention works, dealing with this larger image doesn't just need four times more memory and computing power, it actually needs sixteen times more (since 4 times 4 equals 16). This makes it challenging to work with very high-resolution images, as they require a lot of resources to process. 
Latent diffusion models address the high computational demands of processing large images by using a Variational Auto-Encoder (VAE) to shrink the images into a more manageable size. The idea is that many images have repetitive or unnecessary information. A VAE, after being trained on a lot of data, can compress an image into a much smaller, condensed form. This smaller version still retains the essential features of the original image.

- How are we fusing texts with images since we are using prompts?

We know that during inference time, we can feed in the description of an image we'd like to see and some pure noise as a starting point, and the model does its best to 'denoise' the random input into something that matches the caption.
SD leverages a pre-trained transformer model based on something called [CLIP](https://huggingface.co/learn/computer-vision-course/unit4/multimodal-models/clip-and-relatives/clip). CLIP's text encoder was designed to process image captions into a form that could be used to compare images and text, so it is well suited to the task of creating useful representations from image descriptions. An input prompt is first tokenized (based on a large vocabulary where each word or sub-word is assigned a specific token) and then fed through the CLIP text encoder, producing a 768-dimensional (in the case of SD 1.X) or 1024-dimensional (SD 2.X) vector for each token. To keep things consistent prompts are always padded/truncated to be 77 tokens long, and so the final representation which we use as conditioning is a tensor of shape 77x1024 per prompt.

- How can we add-in good inductive biases? 

Since, we are trying to generate something new(e.g., a realistic Pokemon), we need a way to go beyond the images we have seen before(e.g., an anime Pokemon). That's where U-Net and self-attention come into the picture. Given a noisy version of an image, the model is tasked with predicting the denoised version based on additional clues such as a text description of the image. Ok, how do we actually feed this conditioning information into the U-Net for it to use as it makes predictions? The answer is something called cross-attention. Scattered throughout the U-Net are cross-attention layers. 
Each spatial location in the U-Net can 'attend' to different tokens in the text conditioning, bringing in relevant information from the prompt. 


## How to use `text-to-image`, `image-to-image`, Inpainting Models in Diffusers
This section introduces helpful usecases and how we can perform these tasks using the [Diffusers](https://github.com/huggingface/diffusers) library.
- Steps for `text-to-image` inference
The idea is to pass in the text prompt, which is converted to the output image.

<iframe
	src="https://hysts-controlnet-v1-1.hf.space/"
	frameborder="0"
	width="850"
	height="450">
</iframe>

Using the `diffusers` library you can get `text-to-image` working in 2 steps.

Let's install the `diffusers` library first.
```bash
pip install diffusers
```

We will now initialize the pipeline and pass our prompt inside and infer.
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

- Steps for image-to-image inference
In similar fashion, we can initialize the pipeline, but pass an image and a text prompt instead.
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
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# Load an image to pass to the pipeline:
init_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
)

# Pass a prompt and image to the pipeline to generate an image:
prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
image = pipeline(prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

- Steps for Inpainting
For inpainting pipeline, we need to pass an image, a text prompt, and a mask based on an object in that image, which indicates what to inpaint in the image. 
In this example we also pass a negative prompt to further influence the inference on what we want to avoid.
```python
# Load the pipeline
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# Load the base and mask images:
init_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
)
mask_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"
)

# Create a prompt to inpaint the image with and pass it to the pipeline with the base and mask images:
prompt = (
    "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
)
negative_prompt = "bad anatomy, deformed, ugly, disfigured"
image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    mask_image=mask_image,
).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

### Further Reading
- [Diffusers documentation](https://huggingface.co/docs/diffusers/using-diffusers/pipeline_overview)
- [Diffusers installation](https://huggingface.co/docs/diffusers/installation)