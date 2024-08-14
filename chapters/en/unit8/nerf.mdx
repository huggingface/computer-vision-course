# Neural Radiance Fields (NeRFs)

Neural Radiance Fields are a way of storing a 3D scene within a neural network. This way of storing and representing a scene is often called an implicit representation, since the scene parameters are fully represented by the underlying Multi-Layer Perceptron (MLP). 
(As compared to an explicit representation that stores scene parameters like colour or density explicitly in voxel grids.) 
This novel way of representing a scene showed very impressive results in the task of [novel view synthesis](https://en.wikipedia.org/wiki/View_synthesis), the task of interpolating novel views from camera perspectives that are not in the training set. 
Furthermore, it allows us to store large scenes with a smaller memory footprint than explicit representation, since we merely need to store the weights of our neural network compared to voxel grids, which increase in memory size by a cubic term.

## Short History 📖
The field of NeRFs is relatively young with the first publication by [Mildenhall et al.](https://www.matthewtancik.com/nerf) appearing in 2020. 
Since then, a vast number of papers have been published and fast advancements have been made. 
Since 2020, more than 620 preprints and publications have been released, with more than 250 repositories on GitHub. *(as of Dec 2023, statistics from [paperswithcode.com](https://paperswithcode.com/method/nerf))*.

Since the first formulation of NeRFs requires long training times (up to days on beefy GPUs), there have been a lot of advancements towards faster training and inference. 
An important leap was NVIDIA's [Instant-ngp](https://nvlabs.github.io/instant-ngp/), which was released in 2022. 
While the model architecture used in this approach is similar to existing ones, the authors introduced a novel encoding method that uses trainable hash-tables. 
Thanks to this type of encoding, we can shrink the MLP down significantly without loosing reconstruction quality.
This novel approach was faster to train and query while performing on par quality wise with then state-of-the-art methods. 
[Mipnerf-360](https://jonbarron.info/mipnerf360/), which was also released in 2022, is also worth mentioning. 
Again, the model architecture is the same as for most NeRFs, but the authors introduced a novel scene contraction that allows us to represent scenes that are unbounded in all directions, which is important for real-world applications. 
[Zip-NeRF](https://jonbarron.info/zipnerf/), released in 2023, combines recent advancements like the encoding from [Instant-ngp](https://nvlabs.github.io/instant-ngp/) and the scene contraction from [Mipnerf-360](https://jonbarron.info/mipnerf360/) to handle real-world situation whilst decreasing training times to under an hour. 
*(this is still measured on beefy GPUs to be fair)*.

Since the field of NeRFs is rapidly evolving, we added a section at the end where we will tease the latest research and the possible future direction of NeRFs.

But now enough with the history, let's dive into the intrinsics of NeRFs! 🚀🚀

## Underlying approach (Vanilla NeRF) 📘🔍
The fundamental idea behind NeRFs is to represent a scene as a continuous function that maps a position, \\( \mathbf{x} \in \mathbb{R}^{3} \\), and a viewing direction,  \\( \boldsymbol{\theta} \in \mathbb{R}^{2} \\), to a colour \\( \mathbf{c} \in \mathbb{R}^{3} \\) and volume density \\( \sigma \in \mathbb{R}^{1}\\). 
As neural networks can serve as universal function approximators, we can approximate this continuous function that represents the scene with a simple Multi-Layer Perceptron (MLP) \\( F_{\mathrm{\Theta}} : (\mathbf{x}, \boldsymbol{\theta}) \to (\mathbf{c},\sigma) \\).

A simple NeRF pipeline can be summarized with the following picture:

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/nerf_pipeline.png" alt="nerf_pipeline" />
  <p>Image from: <a href="https://www.matthewtancik.com/nerf">Mildenhall et al. (2020)</a></p>
</div>

**(a)** Sample points and viewing directions along camera rays and pass them through the network.

**(b)** Network output is a colour vector and density value for each sample.

**(c)** Combine the outputs of the network via volumetric rendering to go from discrete samples in 3D space to a 2D image. 

**(d)** Compute the loss and update network gradients via backpropagation to represent scene.

This is overview is very high level, so for a better understanding, let's go into the details of the volume rendering and the used loss function.

**Volume Rendering**

The principles behind the process of volumetric rendering are well established in classical computer graphics pipelines and do not stem from NeRFs. 
What is important for the use case of NeRFs is that this step is **differentiable** in order to allow for backpropagation. The simplest form of volumetric rendering in NeRFs can be formulated as follows:

$$\mathbf{C}(\mathbf{r}) = \int_{t_n}^{t_f}T(t)\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t),\mathbf{d})dt$$

In the equation above, \\( \mathbf{C}(\mathbf{r}) \\) is the expected colour of a camera ray \\( \mathbf{r}(t)=\mathbf{o}+t\mathbf{d} \\), where \\( \mathbf{o} \in \mathbb{R}^{3} \\) is the origin of the camera, \\( \boldsymbol{d} \in \mathbb{R}^{3} \\) is the viewing direction as a 3D unit vector and \\( t \in \mathbb{R}_+ \\) is the distance along the ray. 
\\( t_n \\) and \\( t_f \\) stand for the near and far bounds of the ray, respectively. 
\\( T(t) \\) denotes the accumulated transmittance along ray \\( \mathbf{r}(t) \\) from \\( t_n \\) to \\( t \\).

After discretization, the equation above can be computed as the following sum:

$$\boldsymbol{\hat{C}}(\mathbf{r})=\sum_{i=1}^{N}T_i (1-\exp(-\sigma_i \delta_i)) \mathbf{c}_i\,, \textrm{ where }T_i=\exp \bigg(-\sum_{j=1}^{i-1} \sigma_j \delta_j \bigg)$$

Below, you can see a schematic visualisation of a discretized camera ray in order to get a better sense of the variables from above:

![ray_image](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/nerf_ray_visualisation.png)

**Loss formulation**

As the discretized volumetric rendeing equation is fully differentiable, the weights of the underlying neural network can then be trained using a reconstruction loss on the rendered pixels. 
Many NeRF approaches use a pixel-wise error term that can be written as follows:

$$\mathcal{L}_{\rm recon}(\boldsymbol{\hat{C}},\boldsymbol{C^*}) = \left\|\boldsymbol{\hat{C}}-\boldsymbol{C^*}\right\|^2$$

,where \\( \boldsymbol{\hat{C}} \\) is the rendered pixel colour and \\( \boldsymbol{C}^* \\) is the ground truth pixel colour.

**Additional remarks**

It is very hard to describe the whole NeRF pipeline in detail within a single chapter. 
The explanations above are important to understand the basic concepts and similar if not identical in every NeRF model. 
However, some additional tricks are needed to obtain a well performing model.

First of all, it is necesarry to encode input signals in order to capture high-frequency variations in colour and geometry. 
The practice of encoding inputs before passing them through a neural network is not unique to the NeRF domain but also widely adopted in other ML domains like for example Natural Language Processing (NLP). 
A very simple encoding where we map the inputs to a higher dimensional space, enabling us to capture high frequency variations in scene parameters could look as follows:

```python
import torch
import mediapy as media
import numpy as np


def positional_encoding(in_tensor, num_frequencies, min_freq_exp, max_freq_exp):
    """Function for positional encoding."""
    # Scale input tensor to [0, 2 * pi]
    scaled_in_tensor = 2 * np.pi * in_tensor
    # Generate frequency spectrum
    freqs = 2 ** torch.linspace(
        min_freq_exp, max_freq_exp, num_frequencies, device=in_tensor.device
    )
    # Generate encodings
    scaled_inputs = scaled_in_tensor.unsqueeze(-1) * freqs
    encoded_inputs = torch.cat(
        [torch.sin(scaled_inputs), torch.cos(scaled_inputs)], dim=-1
    )
    return encoded_inputs.view(*in_tensor.shape[:-1], -1)


def visualize_grid(grid, encoded_images, resolution):
    """Helper Function to visualize grid."""
    # Split the grid into separate channels for x and y
    x_channel, y_channel = grid[..., 0], grid[..., 1]
    # Show the original grid
    print("Input Values:")
    media.show_images([x_channel, y_channel], cmap="plasma", border=True)
    # Show the encoded grid
    print("Encoded Values:")
    num_channels_to_visualize = min(
        8, encoded_images.shape[-1]
    )  # Visualize up to 8 channels
    encoded_images_to_show = encoded_images.view(resolution, resolution, -1).permute(
        2, 0, 1
    )[:num_channels_to_visualize]
    media.show_images(encoded_images_to_show, vmin=-1, vmax=1, cmap="plasma", border=True)


# Parameters similar to your NeRFEncoding example
num_frequencies = 4
min_freq_exp = 0
max_freq_exp = 6
resolution = 128

# Generate a 2D grid of points in the range [0, 1]
x_samples = torch.linspace(0, 1, resolution)
y_samples = torch.linspace(0, 1, resolution)
grid = torch.stack(
    torch.meshgrid(x_samples, y_samples), dim=-1
)  # [resolution, resolution, 2]

# Apply positional encoding
encoded_grid = positional_encoding(grid, num_frequencies, min_freq_exp, max_freq_exp)

# Visualize result
visualize_grid(grid, encoded_grid, resolution)
```

The output should look something like the image below:

![encoding](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/nerf_encodings.png)

The second trick worth mentioning is that most methods use smart approaches to sample points in space. 
Essentially, we want to avoid sampling in regions where the scene is empty. 
There are various approaches to concentrate samples in regions that contribute most to the final image, but the most prominent one is to use a second network, often called *proposal network* so that no compute is wasted. 
If you are interested in the inner workings and optimisation of such a *proposal network*, feel free to dig into the publication of [Mipnerf-360](https://jonbarron.info/mipnerf360/), where it was first proposed.

## Train your own NeRF
To get the full experience when training your first NeRF, I recommend taking a look at the awesome [Google Colab notebook from the nerfstudio team](https://colab.research.google.com/github/nerfstudio-project/nerfstudio/blob/main/colab/demo.ipynb). 
There, you can upload images of a scene of your choice and train a NeRF. You could for example fit a model to represent your living room. 🎉🎉

## Current advancements in the field
The field is rapidly evolving and the number of new publications is almost exploding. 
Concerning training and rendering speed, [VR-NeRF](https://vr-nerf.github.io) and [SMERF](https://smerf-3d.github.io) show very promising results. 
We believe that we will soon be able to stream a real-world scene in real-time on an edge device, and this is a huge leap towards a realistic *Metaverse*. 
However, the research in the field of NeRFs is not only focusing on training and inference speed, but encompasses various directions like, Generative NeRFs, Pose Estimation, Deformable NeRFs, Compositionality and many more. 
If you are interested in a curated list of NeRF publications, checkout [Awesome-NeRF](https://github.com/awesome-NeRF/awesome-NeRF).