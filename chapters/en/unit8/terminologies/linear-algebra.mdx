# Basics of Linear Algebra for 3D Data

## Coordinate systems

Most three-dimensional data consists of objects such as points which have a defined position in space, often represented by their three Cartesian coordinates \\([X, Y, Z]\\).

![Axis handedness]( https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/axes_handedness.png)

However, various systems have different conventions for this coordinate system. The most important difference is handedness, which is the relative orientation of the X, Y and Z axes. The easiest way to remember the difference is to point your middle finger inwards, such that your thumb, index, and middle fingers are roughly at right angles to each other. On your left hand, your thumb (X), index finger (Y), and middle finger (Z) form a left-handed coordinate system. Similarly, the fingers of your right hand make a right-handed coordinate system.

In mathematics and physics, a right-handed system is usually used. However, in computer graphics, different libraries and environments have different conventions. Notably, Blender, Pytorch3d and OpenGL (mostly) use right-handed coordinates, whilst DirectX uses left-handed coordinates.Here we will use the right-handed convention, following Blender and NerfStudio.

## Transformations

It is useful to be able to rotate, scale, and translate these coordinates in space. For example, if an object is moving, or if we want to change these coordinates from world coordinates relative to some fixed set of axes, to coordinates relative to our camera.

These transformations can be represented by matrices. Here we'll use `@` to denote matrix multiplication. To allow us to represent translation, rotation and scaling in a consistent manner, we take the three dimensional coordinates \\([x,y,z]\\), and add an extra coordinate \\(w=1\\). These are known as homogeneous coordinates - more generally, \\(w\\) can take any value, and all points on the four-dimensional line \\([wx, wy, wz, w]\\) correspond to the same point \\([x,y,z]\\) in three-dimensional space. However, here, \\(w\\) will always be 1.

Libraries such as [Pytorch3d](https://pytorch3d.org/) provide a range of functions for generating and manipulating transformations.

Yet another convention to note - OpenGL treats positions as column vectors `x` (of shape 4x1), and applies a transformation `M` by pre-multiplying the vector by the matrix (`M @ x`), whereas DirectX and Pytorch3d consider positions as row vectors of shape (1x4), and apply a transformation by post-multiplying the vector by the matrix ( `x @ M` ). To convert between the two conventions we need to take the transpose of the matrix `M.T`. We will show how a cube transforms under different transformation matrices in a few code snippets. For these code snippets, we will use the OpenGL convention.

### Translations

Translations, moving all the points in space by the same distance and direction, can be represented as

$$T = \begin{pmatrix} 1 & 0 & 0 & t_x \\ 0 & 1 & 0 & t_y \\ 0 & 0 & 1 & t_z \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

where \\(t = [t_x,t_y,t_z]\\) is the direction vector to translate all the points.

To try out a translation ourselves, let us first write a little helper function to visualise a cube:

```python
import numpy as np
import matplotlib.pyplot as plt


def plot_cube(ax, cube, label, color="black"):
    ax.scatter3D(cube[0, :], cube[1, :], cube[2, :], label=label, color=color)
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    for line in lines:
        ax.plot3D(cube[0, line], cube[1, line], cube[2, line], color=color)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
```

Now, we can create a cube and pre-multiply it with a translation matrix:

```python
# define 8 corners of our cube with coordinates (x,y,z,w) and w is always 1 in our case
cube = np.array(
    [
        [-1, -1, -1, 1],
        [1, -1, -1, 1],
        [1, 1, -1, 1],
        [-1, 1, -1, 1],
        [-1, -1, 1, 1],
        [1, -1, 1, 1],
        [1, 1, 1, 1],
        [-1, 1, 1, 1],
    ]
)

# translate to follow OpenGL notation
cube = cube.T

# set up figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# plot original cube
plot_cube(ax, cube, label="Original", color="blue")

# translation matrix (shift 1 in positive x and 1 in positive y-axis)
translation_matrix = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])

# translation
translated_cube = translation_matrix @ cube
plot_cube(ax, translated_cube, label="Translated", color="red")
```

The output should look something like this:

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/translation.png" alt="output_translation" />
</div>

### Scaling

Scaling is the process of uniformly increasing or decreasing the size of an object. A scaling transformation is represented by a matrix that multiplies each coordinate by a scale factor. The scaling matrix is given by:

$$S = \begin{pmatrix} s_x & 0 & 0 & 0 \\ 0 & s_y & 0 & 0 \\ 0 & 0 & s_z & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$


Let us try the following example of a scaling our cube by a factor of 2 along the X-axis and 0.5 along the Y-axis.

```python
# set up figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# plot original cube
plot_cube(ax, cube, label="Original", color="blue")

# scaling matrix (scale by 2 along x-axis and by 0.5 along y-axis)
scaling_matrix = np.array([[2, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


scaled_cube = scaling_matrix @ cube

plot_cube(ax, scaled_cube, label="Scaled", color="green")
```

The output should look something like this:

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/scaling.png" alt="output_scaling" />
</div>

### Rotations

Rotations around an axis are another commonly used transformation. There are a number of different ways of representing rotations, including Euler angles and quaternions, which can be very useful in some applications. Again, libraries such as Pytorch3d include a wide range of functionalities for performing rotations. However, as a simple example, we will just show how to construct rotations about each of the three axes.

- Rotation around the X-axis:

$$ R_x(\alpha) = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos \alpha & -\sin \alpha & 0 \\ 0 & \sin \alpha & \cos \alpha & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} $$

A little example for a positive 20 degree roation around the X-axis is given below:

```python
# set up figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# plot original cube
plot_cube(ax, cube, label="Original", color="blue")

# rotation matrix: +20 deg around x-axis
angle = 20 * np.pi / 180
rotation_matrix = np.array(
    [
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1],
    ]
)


rotated_cube = rotation_matrix @ cube

plot_cube(ax, rotated_cube, label="Rotated", color="orange")
```

The output should look something like this:

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/rotation.png" alt="output_rotation" />
</div>

- Rotation around the Y-axis:

$$ R_y(\beta) = \begin{pmatrix} \cos \beta & 0 & \sin \beta & 0 \\ 0 & 1 & 0 & 0 \\ -\sin \beta & 0 & \cos \beta & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} $$

We are sure you can use the example snippet above and figure out how to implement a rotation around the Y-axis.😎😎

- Rotation around the Z-axis

$$ R_z(\beta) = \begin{pmatrix} \cos \beta & -\sin \beta & 0 & 0 \\ \sin \beta & \cos \beta & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} $$

Again, can you use the last code snippet and implement a rotation around the Z-axis❓

Note that the standard convention is that a positive rotation angle corresponds to an anti-clockwise rotation when the axis of rotation is pointing toward the viewer. Also note that in most libraries the cosine function requires the angle to be in radians. To convert from
degrees to radians, multiply by \\( pi/180\\).

### Combining transformations

Multiple transformations can be combined by multiplying together their matrices. Note that the order that matrices are multiplied matters - with the matrices being applied right to left. To make a matrix that applies the transforms P, Q, and R, in that order, the composite transformation is given by \\( X = R @ Q @ P\\).

If we want to do first the translation, then the rotation, and then the scaling that we did above in one operation, it looks as follows:

```python
# set up figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# plot original cube
plot_cube(ax, cube, label="Original", color="blue")

# combination of transforms
combination_transform = rotation_matrix.dot(scaling_matrix.dot(translation_matrix))
final_result = combination_transform.dot(cube)
plot_cube(ax, final_result, label="Combined", color="violet")
```

The output should look something like the following.

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/combined.png" alt="output_combined" />
</div>