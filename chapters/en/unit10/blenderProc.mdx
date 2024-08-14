# Using a 3D Renderer to Generate Synthetic Data

When creating computer-generated images to use as synthetic training data, ideally we want the images to look as realistic as possible.
Physically Based Renderers (PBR) such as [Blender Cycles](https://www.blender.org)
or [Unity](https://unity.com) help to create images that are super realistic and look and feel just like they do in the real world.

Imagine you're creating an image of a shiny apple. Now, when you color that apple, you want it to look realistic, right?
That's where something called PBR comes in.

![apple](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/apple.jpg)

Okay, let's break it down:

_Colors and Light:_

- When light shines on objects, it interacts with them in different ways. PBR tries to figure out and simulate this interaction.
- Think about how light hits the apple. Some parts may be brighter where the light directly hits, and some parts might be darker
  where the light is blocked or doesn't reach as much.

_Materials:_

- Different materials react to light differently. For example, a shiny metallic surface reflects light more than a soft, matte fabric.
- PBR takes into account the material of an object, so if you're rendering a metal vase, it will reflect light differently than a fluffy teddy bear.

_Textures:_

- PBR uses textures to add details like bumps, scratches, or tiny grooves on the surface of objects. This makes things look more real because, in the real world, very few things are perfectly smooth.

_Realism:_

- PBR aims to make things look as close to real life as possible. It does this by considering how light behaves in reality, how different materials interact with light, and how surfaces have small imperfections.

_Layers of Light:_

- Imagine you're looking at a glass of water. PBR will try to simulate the way light passes through the water and how it might distort what you see.
- It considers how multiple layers of how light interact with different parts of an object, making the rendered image more realistic.

PBR also simplifies the workflow. Instead of manually tweaking many parameters to get the right look,
you can use a set of standardized materials and lighting models.
This makes the process more intuitive and user-friendly.

Now, think about training AI models like those used in computer vision.
If you're teaching a computer to recognize objects in images, it's beneficial to have a diverse set of
images that closely mimic real-world scenarios. PBR helps in generating synthetic data that looks so real that it can be used
to train computer vision models effectively.

There are several 3D rendering engines that you can use for PBR, including [Blender Cycles](https://www.blender.org)
or [Unity](https://unity.com). We are going to focus on Blender because it is open source and there are a lot of resources about Blender.

## Blender

Blender is a powerful, open-source 3D computer graphics software used for creating animated films, visual effects, art, 3D games, and more.
It encompasses a wide range of features, making it a versatile tool for artists, animators, and developers.
Let's start off by walking through a basic example of rendering a synthetic image of an elephant.

Here are the essential steps:

- Create the elephant model. The one shown below was created with the [Metascan](https://metascan.ai) app using Photogrammetry.
  Photogrammetry is a way of turning regular photos into a 3D model. It's like taking a bunch of pictures of your toy from different angles
  and then using those pictures to make a computer version of it.
- Create the background - this was a multi-step process. See [here](https://github.com/kfahn22/Synthetic-Data-Creation-in-Blender/tree/main/BACKGROUND)
  for a more detailed explanation.
- Adjust the lighting and camera positions.
- Fix the location and rotation of the elephant so that it fits within the frame (or camera view).

Here is the elephant image generated in Blender:

![elephant image](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-PBR/rendered_elephant.png)

It is not completely photorealistic, but probably close enough to train a model to monitor elephant populations. Of course, to do
that we need to create a large dataset of synthetic elephant images! You can use the Blender python environment
[bpy](https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html) to render
a large number of images with the location and rotation of the elephant randomized.
You can also use a script to help with segmentation, depth, normal, and pose estimation.

Great! How do we get started?

Unfortunately, there is a pretty steep learning curve associated with Blender. None of the steps are too complicated,
but wouldn't it be nice if we could render the dataset without trying to figure all of this out?
Luckily for us, there is a library called BlenderProc that has all the scripts we need to render realistic synthetic data
and annotations and it is built on top of Blender.

## BlenderProc

The BlenderProc pipeline was introduced in [BlenderProc](https://arxiv.org/abs/1911.01911), Denninger, et. al. and is a modular pipeline built on top of [Blender](https://www.blender.org/).
It can be used to generate images in a variety of use cases, including segmentation, depth, normal and pose estimation.

It is specifically created to help in the generation of realistic looking images for the training of convolutional neural networks.  
 It has the following properties which make it a great choice for synthetic data generation:

- Procedural Generation: Enables the automated creation of complex 3D scenes with variations using procedural techniques.
- Simulation: Supports the integration of simulations, including physics simulations, to enhance realism.
- Large-Scale Generation: Designed to handle large-scale scene generation efficiently, making it suitable for diverse applications.
- Automation and Scalability:
  - Scripting: Allows users to automate the generation process by employing python scripts to tailor BlenderProc to their specific needs and configure parameters.
  - Parallel Processing: Supports parallel processing for scalability, making it efficient for generating a large number of scenes.

You can install BlenderProc via pip:

```bash
pip install blenderProc
```

Alternately, you can clone the official [BlenderProc repository](https://github.com/DLR-RM/BlenderProc) from GitHub using Git:

```bash
git clone https://github.com/DLR-RM/BlenderProc
```

BlenderProc must be run inside the blender python environment (bpy), as this is the only way to access the Blender API.

```bash
blenderproc run <your_python_script>
```

You can check out this notebook to try BlenderProc in Google Colab, demos the basic examples provided [here](https://github.com/DLR-RM/BlenderProc/tree/main/examples/basics).
Here are some images rendered with the basic example:

![colors](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-PBR/colors.png)
![normals](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-PBR/normals.png)
![depth](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-PBR/depth.png)

## Blender Resources

- [User Manual](https://docs.blender.org/manual/en/latest/0)
- [Awesome-blender -- Extensive list of resources](https://awesome-blender.netlify.app)
- [Blender Youtube Channel](https://www.youtube.com/@BlenderOfficial)

### The following video explains how to render a 3D syntehtic dataset in Blender:

<Youtube id="E1Pqpfg5kSo" />

### The following video explains how to create a 3D object using Photogrammetry:

<Youtube id="Pcqokf3PG_4" />

## Papers / Blogs

- [Developing digital twins of multi-camera metrology systems in Blender](https://iopscience.iop.org/article/10.1088/1361-6501/acc59e/pdf_)
- [Generate Depth and Normal Maps with Blender](https://www.saifkhichi.com/blog/blender-depth-map-surface-normals)
- [Object detection with synthetic training data](https://medium.com/rowden/object-detection-with-synthetic-training-data-f6735a5a34bc)

## BlenderProc Resources

- [BlenderProc Github Repo](https://github.com/DLR-RM/BlenderProc)
- [BlenderProc: Reducing the Reality Gap with Photorealistic Rendering](https://elib.dlr.de/139317/1/denninger.pdf)
- [Documentation](https://dlr-rm.github.io/BlenderProc/)

### The following video provides an overview of the BlenderProc pipeline:

<Youtube id="1AvY_iS6xQA" />

## Papers

- [3D Menagerie: Modeling the 3D Shape and Pose of Animals]()
- [Fake It Till You Make It: Face analysis in the wild using synthetic data alone]()
- [Object Detection and Autoencoder-Based 6D Pose Estimation for Highly Cluttered Bin Picking](https://arxiv.org/pdf/2106.08045.pdf)
- [Learning from Synthetic Animals](https://arxiv.org/abs/1912.08265)
