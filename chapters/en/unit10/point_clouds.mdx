# Introduction

## What is a point cloud?

A point cloud is a collection of individual points, each representing a sample of a surface within a three-dimensional space denoted by [x, y, z] coordinates. Beyond their spatial coordinates, these points often carry additional attributes like normals, RGB color, albedo, and Bidirectional Reflectance Distribution Function (BRDF).

Here, albedo is the measure of how much light a surface reflects. It's essentially the ratio of reflected light to the incident light that strikes the surface. In simpler terms, it describes how much of the incoming light is bounced back. A high albedo indicates a surface that reflects a lot of light, such as snow, while a low albedo suggests a surface that absorbs more light, like asphalt.

The BRDF is a function that describes how light is scattered or reflected at an opaque surface. It details the way light is reflected at an intersection point on a surface, considering the incoming light direction and the outgoing direction. It provides a mathematical description of the surface's reflective properties, including factors like glossiness, roughness, and the distribution of reflected light over different angles.
These attributes serve crucial roles in various applications such as modeling, rendering, and scene comprehension.

While the concept of point cloud data isn't new and has been integral in fields like graphics and physics simulation for many years, its significance has notably surged due to two key trends.
Firstly, the widespread availability of cost-effective and user-friendly point cloud acquisition devices has significantly increased accessibility.

Augmented Reality and autonomous vehicles have further underscored their relevance in today's technological landscape.

![An Example of Point Clouds in Action](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/point_cloud_example.jpeg)

**Now that we know what a Point Cloud is, what can we do with them?**

The 3D Point Data is mainly used in self-driving capabilities, but now other AI models using computer vision like drones and robots are also using LiDAR for better visual perception. LiDAR is a remote sensing process that collects measurements used to create 3D models and maps of objects and environments. Using ultraviolet, visible, or near-infrared light, LiDAR gauges spatial relationships and shapes by measuring the time it takes for signals to bounce off objects and return to the scanner.

## Generation and Data Representation

We will be using the python library [point-cloud-utils](https://github.com/fwilliams/point-cloud-utils), and [open-3d](https://github.com/isl-org/Open3D), which can be installed by:

```bash
pip install point-cloud-utils
```

We will be also using the python library open-3d, which can be installed by:

```bash
pip install open3d
```

OR a Smaller CPU only version:

```bash
pip install open3d-cpu
```

Now, first we need to understand the formats in which these point clouds are stored in, and for that, we need to look at mesh cloud.

**Why?**

- `point-cloud-utils` supports reading common mesh formats (PLY, STL, OFF, OBJ, 3DS, VRML 2.0, X3D, COLLADA).
- If it can be imported into [MeshLab](https://github.com/cnr-isti-vclab/meshlab), we can read it! (from their readme)

The type of file is inferred from its file extension. Some of the extensions supported are:

** PLY (Polygon File Format) **

- A simple PLY object consists of a collection of elements for representation of the object. It consists of a list of (x,y,z) triplets of a vertex and a list of faces that are actually indices into the list of vertices.
- Vertices and faces are two examples of elements and the majority of the PLY file consists of these two elements.
- New properties can also be created and attached to the elements of an object, but these should be added in such a way that old programs do not break when these new properties are encountered.

** STL (Standard Tessellation Language) **

- This format approximates the surfaces of a solid model with triangles.
- These triangles are also known as facets, where each facet is described by a perpendicular direction and three points representing the vertices of the triangle.
- However, these files have no description of Color and Texture.

** OFF (Object File Format) **

- Object File Format (.OFF) files are used to represent the geometry of a model by specifying the polygons of the model's surface. The polygons can have any number of vertices.
- It supports ASCII text versions of objects for the purpose of interchange, and binary versions for efficiency of reading and writing
- It is also refered to as the .obj format.

** 3DS (3D Studio) **

- A file with .3ds extension represents the 3D Studio mesh file format used by Autodesk 3D Studio.
- The 3DS format utilizes a binary file structure, enabling faster and smaller file sizes compared to text-based formats, with data organized into chunks within the file.
- These Chunks store the shapes, lighting, and viewing information that together represent the three-dimensional scene.

** X3D (Extensible 3D Graphics) **

- X3D is an XML based 3D graphics file format for presentation of 3D information. It is a modular standard and is defined through several ISO specifications.
- The format supports vector and raster graphics, transparency, lighting effects, and animation settings including rotations, fades, and swings.
- X3D has the advantage of encoding color information (unlike STL) that is used during printing the model on a color 3D printer.

** DAE (Digital Asset Exchange) **

- This is an XML schema which is an open standard XML schema, from which DAE files are built.
- This file format is based on the COLLADA (COLLAborative Design Activity) XML schema which is an open standard XML schema for the exchange of digital assets among graphics software applications.
- The format's biggest selling point is its compatibility across multiple platforms.
- COLLADA files aren't restricted to one program or manufacturer. Instead, they offer a standard way to store 3D assets.
