# Introduction
This section explains how stereo vision works and how it can be used to find the 3-dimensional structure of surrounding objects. Stereo vision involves capturing two or more images of the same scene from varying positions and viewpoints. These images can be obtained using multiple cameras or by repositioning the same camera.

## Problem Statement 
Let's understand the problem statement of finding the 3D structure of objects by understanding the geometry of image formation. As shown in Figure 1, we have a point P in 3D with x, y, z coordinates. Point P gets projected to the camera's image plane via the pinhole. This can also be viewed as projecting a 3D point to a 2D image plane. 

Now, let's say we are given this 2D image and the location of the pixel coordinates of point P in this image. We want to find the 3D coordinates of point P. Is this possible? Is point P unique, or are there other 3D points that also map to the same pixel coordinates as point P? Answer is that all 3D points that lie on the line joining point P, and the pinhole will map to the same pixel coordinates in the 2D image plane. 

We aim to solve the problem of determining the 3D structure of objects. In our problem statement, we can represent an object in 3D as a set of 3D points. Finding the 3D coordinates of each of these points helps us determine the 3D structure of the object.

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/3d_stereo_vision_images/image_formation_single_camera.png?download=true" alt="Figure 1: Image formation using single camera" />
  <p>Figure 1: Image formation using single camera</p>
</div>

## Solution 
Let's assume we are given the following information:

1. Single image of a scene point P
2. Pixel coordinates of point P in the image
3. Position and orientation of the camera used to capture the image. For simplicity, we can also place an XYZ coordinate system at the location of the pinhole, with the z-axis perpendicular to the image place and the x-axis, and y-axis parallel to the image plane like in Figure 1.
4. Internal parameters of the camera, such as focal length and location of principal point. The principal point is where the optical axis intersects the image plane. Its location in the image plane is usually denoted as (Ox,Oy).

With the information provided above, we can find a 3D line that originates from the pixel coordinates of point P (the projection of point P in the image plane), passes through the pinhole, and extends to infinity. Based on the principles of image formation geometry, we can conclude that point P must exist somewhere along this line.

1. Initially (without an image) point P could have been present anywhere in the 3D space. 
2. Using a single image, we reduced possible locations of point P to a single line in 3D. 
3. Now, let's consider whether we can further narrow down the potential locations to pinpoint the precise location of point P on this 3D line. 
4. Imagine moving the camera to a different position. Let the coordinate system remain fixed at the previous position. The 3D line we found also remains the same and point P still lies somewhere on this line.
5. From this new location of the camera, capture another image of the same scene point P. Once more, utilizing the pixel coordinates of point P within this new image and considering the updated location of the camera pinhole, find the 3D line on which point P must lie.
6. Now we have 2 lines in 3D and point P lies somewhere on both of these lines. So, point P must lie on the intersection of these 2 lines. 

Given 2 lines in 3D, there are are three possibilities for their intersection:

1. Intersect at exactly 1 point
2. Intersect at infinite number of points
3. Do not intersect

If both images (with original and new camera positions) contain point P, we can conclude that the 3D lines must intersect at least once and that the intersection point is point P. Furthermore, we can envision infinite points where both lines intersect only if the two lines are collinear. This is achievable if the pinhole at the new camera position lies somewhere on the original 3D line. For all other positions and orientations of the new camera location, the two 3D lines must intersect precisely at one point, where point P lies. 

Therefore, using 2 images of the same scene point P, known positions and orientations of the camera locations, and known internal parameters of the camera, we can precisely find where point P lies in the 3D space.

## Simplified Solution 
Since there are many different positions and orientations for the camera locations which can be selected, we can select a location that makes the math simpler, less complex, and reduces computational processing when running on a computer or an embedded device. One configuration that is popular and generally used is shown in Figure 2. We use 2 cameras in this configuration, which is equivalent to a single camera for capturing 2 images from 2 different locations.

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/3d_stereo_vision_images/image_formation_simple_stereo.jpg?download=true" alt="Figure 2: Image formation using 2 cameras" />
  <p>Figure 2: Image formation using 2 cameras</p>
</div>

1. Origin of the coordinate system is placed at the pinhole of the first camera which is usually the left camera.
2. Z axis of the coordinate system is defined perpendicular to the image plane. 
3. X and Y axis of the coordinate system are defined parallel to the image plane.  
4. We also have X and Y directions in a 2D image. X is the horizontal direction and Y is the vertical direction. We will refer to these directions in the image plane as u and v respectively. Therefore, pixel coordinates of a point are defined using (u,v) values.  
5. X axis of the coordinate system is defined as the u direction / horizontal direction in the image plane.
6. Similarly Y axis of the coordinate system is defined as the v direction / vertical direction in the image plane. 
7. Second camera (more precisely the pinhole of the second camera) is placed at a distance b called baseline in the positive x direction to the right of the first camera. Therefore, x,y,z coordinates of pinhole of second camera are (b,0,0).
5. Image plane of the second camera is oriented parallel to the image plane of the first camera.  
6. u and v directions in the image plane of second/right camera are aligned with the u and v directions in the image plane of the first/left camera.
7. Both left and right cameras are assumed to have the same intrinsic parameters like focal length and location of principal point.

With the above configuration in place, we have the below equations which map a point in 3D to the image plane in 2D. 

1. Left camera  
    1.  \\(u\_left = f\_x * \frac{x}{z} + O\_x\\)   
    2.  \\(v\_left = f\_y * \frac{y}{z} + O\_y\\)   
  
2. Right camera   
    1.  \\(u\_right = f\_x * \frac{x-b}{z} + O\_x\\)   
    2.  \\(v\_right = f\_y * \frac{y}{z} + O\_y\\)    

Different symbols used in above equations are defined below:    
*  \\(u\_left\\), \\(v\_left\\) refer to pixel coordinates of point P in the left image.
*  \\(u\_right\\),  \\(v\_right\\) refer to pixel coordinates of point P in the right image.   
*  \\(f\_x\\) refers to the focal length (in pixels) in x direction and \\(f\_y\\) refers to the focal length (in pixels) in y direction. Actually, there is only 1 focal length for a camera which is the distance between the pinhole (optical center of the lens) to the image plane. However, pixels may be rectangular and not perfect squares, resulting in different fx and fy values when we represent f in terms of pixels.    
*  x, y, z are 3D coordinates of the point P (any unit like cm, feet, etc can be used).
*  \\(O\_x\\)  and  \\(O\_y\\)  refer to pixel coordinates of the principal point.   
*  b is called the baseline and refers to the distance between the left and right cameras. Same units are used for both b and x,y,z coordinates (any unit like cm, feet, etc can be used).   
  
We have 4 equations above and 3 unknowns - x, y and z coordinates of a 3D point P. Intrinsic camera parameters - focal lengths and principal point are assumed to be known. Equations 1.2 and 2.2 indicate that the v coordinate value in the left and right images is the same. 

3.  \\(v\_left = v\_right\\)    

Using equations 1.1, 1.2 and 2.1 we can derive the x,y,z coordinates of point P.    
     
4.  \\(x = \frac{b * (u\_left - O\_x)}{u\_left - u\_right}\\)    
5.  \\(y = \frac{b * f\_x * (v\_left - O\_y)}{ f\_y * (u\_left - u\_right)}\\)    
6.  \\(z = \frac{b * f\_x}{u\_left - u\_right}\\)     

Note that the x and y values above concern the left camera since the origin of the coordinate system is aligned with the left camera. The above equations show that we can find 3D coordinates of a point P using its 2 images captured from 2 different camera locations. z value is also referred to as the depth value. Using this technique, we can find the depth values for different pixels within an image and their real-world x and y coordinates. We can also find real-world distances between different points in an image. 

## Demo 
### Setup 
We'll work through an example, capture some images, and perform some calculations to find out if our above assumptions and math work out! For capturing the images we'll use a hardware known as OAK-D Lite (OAK stands for OpenCV AI Kit). This device has 3 cameras - left and right mono (black and white) and a center color cameras. We'll use the left and right mono cameras for our experiment. A regular smartphone camera could also be used, but OAK-D lite has some advantages listed below.

* Intrinsic camera parameters like focal length and location of principal point are known for an OAK-D Lite since the device comes pre-calibrated, and these parameters can be read from the device using its Python API. For a smartphone camera, intrinsic parameters need to be determined and could be found by performing camera calibration or sometimes present in the metadata of the image captured using the smartphone. 
* One of the main assumptions above is that the position and orientation of the left and right cameras are known. Using a smartphone camera, it may be difficult to determine this information or additional calibration may be required.  On the other hand, for an OAK-D Lite device, the position and orientation of the left and right cameras are fixed, known, pre-calibrated, and very similar to the geometry of the simplified solution mentioned above. Although some post-processing/image rectification detailed below on the raw images is still required.

### Raw Left and Right Images
The left and right cameras in OAK-D Lite are oriented similarly to the geometry of the simplified solution detailed above. The baseline distance between the left and right cameras is 7.5cm. Left and right images of a scene captured using this device are shown below. The figure also shows these images stacked horizontally with a red line drawn at a constant height (i.e. at a constant v value ). We'll refer to the horizontal x-axis as u and the vertical y-axis as v. 

Raw Left Image
<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/3d_stereo_vision_images/unrectified_left_frame.jpg?download=true" alt="Raw Left Image" />
</div>

Raw Right Image 
<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/3d_stereo_vision_images/unrectified_right_frame.jpg?download=true" alt="Raw Right Image" />
</div>

Raw Stacked Left and Right Images 
<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/3d_stereo_vision_images/unrectified_stacked_frames.jpg?download=true" alt="Raw Stacked Left and Right Images" />
</div>

Let's focus on a single point - the top left corner of the laptop. As per equation 3 above,  \\(v\_left = v\_right\\)  for the same point in the left and right images. However, notice that the red line, which is at a constant v value, touches the top-left corner of the laptop in the left image but misses this point by a few pixels in the right image. There are two main reasons for this discrepancy:

* The intrinsic parameters for the left and right cameras are different. The principal point for the left camera is at (319.13, 233.86), whereas it is (298.85, 245.52) for the right camera. The focal length for the left camera is 450.9, whereas it is 452.9 for the right camera. The values of fx are equal to fy for both the left and right cameras. These intrinsic parameters were read from the device using it's python API and could be different for different OAK-D Lite devices.
* Left and right camera orientations differ slightly from the geometry of the simplified solution detailed above. 

### Rectified Left and Right Images 
We can perform image rectification/post-processing to correct for differences in intrinsic parameters and orientations of the left and right cameras. This process involves performing 3x3 matrix transformations. In the OAK-D Lite API, a stereo node performs these calculations and outputs the rectified left and right images. Details and source code can be viewed [here](https://github.com/luxonis/depthai-experiments/blob/master/gen2-stereo-on-host/main.py). In this specific implementation, correction for intrinsic parameters is performed using intrinsic camera matrices, and correction for orientation is performed using rotation matrices(part of calibration parameters) for the left and right cameras. The rectified left image is transformed as if the left camera had the same intrinsic parameters as the right one. Therefore, in all our following calculations, we'll use the intrinsic parameters for the right camera i.e. focal length of 452.9 and principal point at (298.85, 245.52). In the rectified and stacked images below, notice that the red line at constant v touches the top-left corner of the laptop in both the left and right images.

Rectified Left Image
<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/3d_stereo_vision_images/rectified_left_frame.jpg?download=true" alt="Rectified Left Image" />
</div>

Rectified Right Image 
<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/3d_stereo_vision_images/rectified_right_frame.jpg?download=true" alt="Rectified Right Image" />
</div>

Rectified and Stacked Left and Right Images 
<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/3d_stereo_vision_images/rectified_stacked_frames.jpg?download=true" alt="Rectified and Stacked Left and Right Images" />
</div>

Let's also overlap the rectified left and right images to see the difference. We can see that the v values for different points remain mostly constant in the left and right images. However, the u values change, and this difference in the u values helps us find the depth information for different points in the scene, as shown in Equation 6 above. This difference in 'u' values \\(u\_left - u\_right\\) is called disparity, and we can notice that the disparity for points near the camera is greater compared to points further away. Depth z and disparity  \\(u\_left - u\_right\\)  are inversely proportional, as shown in equation 6.

Rectified and Overlapped Left and Right Images 
<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/3d_stereo_vision_images/rectified_overlapping_frames.jpg?download=true" alt="Rectified and Overlapped Left and Right Images" />
</div>

### Annotated Left and Right Rectified Images
Let's find the 3D coordinates for some points in the scene. A few points are selected and manually annotated with their (u,v) values, as shown in the figures below. Instead of manual annotations, we can also use template-based matching, feature detection algorithms like SIFT, etc for finding corresponding points in left and right images. 

Annotated Left Image 
<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/3d_stereo_vision_images/annotated_left_img.jpg?download=true" alt="Annotated Left Image" />
</div>

Annotated Right Image 
<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/3d_stereo_vision_images/annotated_right_img.jpg?download=true" alt="Annotated Right Image" />
</div>

### 3D Coordinate Calculations  
Twelve points are selected in the scene, and their (u,v) values in the left and right images are tabulated below. Using equations 4, 5, and 6, (x,y,z) coordinates for these points are also calculated and tabulated below. X and Y coordinates concerning the left camera, and the origin is at the left camera's pinhole (or optical center of the lens). Therefore, 3D points left and above the pinhole have negative X and Y values, respectively.

| point    |   \\(u\_left\\)  |   \\(v\_left\\)  |   \\(u\_right\\)  |   \\(v\_right\\)  |   depth/z(cm)  |   \\(x\_wrt\_left\\)|   \\(y\_wrt\_left\\)  |
|:--------:|:---------:|:---------:|:----------:|:----------:|:--------------:|:-----------------:|:-----------------:|
| pt1     |      138 |      219 |       102 |       219 |         94.36 |           -33.51 |            -5.53 |
| pt2     |      264 |      216 |       234 |       217 |        113.23 |            -8.72 |            -7.38 |
| pt3     |      137 |      320 |       101 |       321 |         94.36 |           -33.72 |            15.52 |
| pt4     |      263 |      303 |       233 |       302 |        113.23 |            -8.97 |            14.37 |
| pt5     |      307 |      211 |       280 |       211 |        125.81 |             2.26 |            -9.59 |
| pt6     |      367 |      212 |       339 |       212 |        121.32 |            18.25 |            -8.98 |
| pt7     |      305 |      298 |       278 |       298 |        125.81 |             1.71 |            14.58 |
| pt8     |      365 |      299 |       338 |       299 |        125.81 |            18.37 |            14.86 |
| pt9     |      466 |      225 |       415 |       225 |         66.61 |            24.58 |            -3.02 |
| pt10    |      581 |      225 |       530 |       226 |         66.61 |            41.49 |            -3.02 |
| pt11    |      464 |      387 |       413 |       388 |         66.61 |            24.29 |            20.81 |
| pt12    |      579 |      388 |       528 |       390 |         66.61 |            41.2  |            20.95 |


### Dimension Calculations and Accuracy 
We can also compute 3D distances between different points using their (x,y,z) values using the formulae \\(distance = \sqrt{(x\_2 - x\_1)^2 + (y\_2 - y\_1)^2 + (z\_2 - z\_1)^2}\\). Computed distances between some of the points are tabulated below along with their actual measured values. Percentage error \\(( \frac{(actual-measured) * 100}{actual}\\)) is also computed and tabulated. Notice that the calculated and actual values match very well with a percentage error of 1.2% or less. 

| dimension    |   calculated(cm)  |   actual(cm)  |       % error       |
|:------------:|:---------------:|:-------------:|:-------------------:|
| d1(1-2)     |           31.2 |         31.2 |               0    |
| d2(1-3)     |           21.1 |         21.3 |               0.94 |
| d3(5-6)     |           16.6 |         16.7 |               0.6  |
| d4(5-7)     |           24.2 |         24   |               0.83 |
| d5(9-10)    |           16.9 |         16.7 |               1.2  |
| d6(9-11)    |           23.8 |         24   |               0.83 |

Calculated Dimension Results  
![Calculated Dimension Results](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/3d_stereo_vision_images/calculated_dim_results.png?download=true)

## Conclusion
1. In summary, we learned how stereo vision works, the equations used to find the real-world coordinates (x, y, z) of a point P given its two images captured from different viewpoints, and compared theoretical values with experimental results. 
2. We assumed that the intrinsic parameters - focal length and principal point of the cameras - are known, along with their position and orientation information. This is also referred to as calibrated stereo vision.
3. Interestingly, it is also possible to find the 3D coordinates of a point, P, if the position and orientation of the cameras are unknown. In fact, the position and orientation of the cameras with respect to each other can be found using the images themselves. This is referred to as uncalibrated stereo vision!

## References 
1. 3D Reconstruction - Multiple Viewpoints [Coursera](https://www.coursera.org/learn/3d-reconstruction-multiple-viewpoints)
2. Stereo Vision and Depth Estimation using OpenCV AI Kit [LearnOpenCV](https://learnopencv.com/stereo-vision-and-depth-estimation-using-opencv-ai-kit/)
3. OAK-D Lite [Luxonics](https://docs.luxonis.com/projects/hardware/en/latest/pages/DM9095/)