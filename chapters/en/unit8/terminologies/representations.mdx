# Representations for 3D Data

Depending on the application, one of a number of different representations for 3D data might be used. 
Here we'll outline some of the more common ones.

## Point Clouds

Point clouds represent data by lists of points in 3D space, with their coordinates and perhaps with other associated features. These can be distributed just on the surface of an object, or spread throughout its interior. Point clouds are commonly generated through 3D scanning techniques, such as LIDARs. They lack information about connectivity, which can make it difficult to determine the surface of the object and its topology.

## Meshes

Meshes are commonly used in computer graphics, representing the surfaces of objects as collections of (usually) triangular faces, connecting vertices in three-dimensional space. Additional information such as normals, colors, or texture coordinates can be associated with either the vertices or the faces. Especially when a texture is used, these provide a very efficient method for storing solid objects, and are commonly used in games and in other computer graphics applications.

The Python `trimesh` package contains many useful functions for working with mesh data, in particularly loading and saving common data formats.

## Volumetric Data

Volumetric data is commonly used to encode information about transparent objects, such as clouds and fire. Fundamentally, it takes the form of a function \\( f(x,y,z) \\) mapping positions in space to a density, color, and possibly other attributes. One simple method of representing such data is as a volumetric grid, where the data at each point is found by trilinear interpolation from the eight corners of the voxel containing it.

As will be seen later in the NeRF chapter, volumetric representations can also be effectively used to represent solid objects. More sophisticated representations can also be used, such as a small MLP, or complex hash-grids such as in InstantNGP.

## Implicit Surfaces

Sometimes the flexibility of a volumetric representation is desirable, but the surface of the object itself is of interest. Implicit surfaces are like volumetric data, but where the function \\( f(x,y,z) \\) maps each point in space to a single number, and where the surface is the zero of this function. For computational efficiency, it can be useful to require that this function is actually a signed distance function (SDF), where the function \\( f(x,y,z) \\) indicates the sortest distance to the surface, with positive values outside the object and negative values inside (this sign is a convention and may vary). Maintaining this constraint is more difficult, but it allows intersections between straight lines and the surface to be calculated more quickly, using an algorithm known as sphere tracing.