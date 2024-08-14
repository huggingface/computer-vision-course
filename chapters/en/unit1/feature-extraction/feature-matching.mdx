# Feature Matching

How can we match detected features from one image to another? Feature matching involves comparing key attributes in different images to find similarities. Feature matching is useful in many computer vision applications, including scene understanding, image stitching, object tracking, and pattern recognition.

## Brute-Force Search

Imagine you have a giant box of puzzle pieces, and you're trying to find a specific piece that fits into your puzzle. This is similar to searching for matching features in images. Instead of having any special strategy, you decide to check every piece, one by one until you find the right one. This straightforward method is a brute-force search. The advantage of brute force is its simplicity. You don't need any special tricks – just patience. However, it can be time-consuming, especially if there are a lot of pieces to check. In the context of feature matching, this brute force approach is akin to comparing every pixel in one image to every pixel in another to see if they match. It's exhaustive and it might take a lot of time, especially for large images.

Now that we have an intuitive idea of how brute-force matches are found, let's dive into the algorithms. We are going to use the descriptors that we learned about in the previous chapter to find the matching features in two images.

First install and load libraries.

```bash
!pip install opencv-python
```

```python
import cv2
import numpy as np
```

**Brute Force with SIFT**

Let's start by initializing SIFT detector.

```python
sift = cv2.SIFT_create()
```

Find the keypoints and descriptors with SIFT.

```python
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
```

Find matches using k nearest neighbors.

```python
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
```

Apply ratio test to threshold the best matches.

```python
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
```

Draw the matches.

```python
img3 = cv2.drawMatchesKnn(
    img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/feature-extraction-feature-matching/SIFT.png" alt="SIFT">
</div>

**Brute Force with ORB (binary) descriptors**

Initialize the ORB descriptor.

```python
orb = cv2.ORB_create()
```

Find keypoints and descriptors.

```python
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
```

Because ORB is a binary descriptor, we find matches using [Hamming Distance](https://www.geeksforgeeks.org/hamming-distance-two-strings/),
which is a measure of the difference between two strings of equal length.

```python
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
```

We will now find the matches.

```python
matches = bf.match(des1, des2)
```

We can sort them in the order of their distance like the following.

```python
matches = sorted(matches, key=lambda x: x.distance)
```

Draw first n matches.

```python
img3 = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    matches[:n],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
```

**Fast Library for Approximate Nearest Neighbors (FLANN)**

FLANN was proposed in [Fast Approximate Nearest Neighbors With Automatic Algorithm Configuration](https://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_visapp09.pdf) by Muja and Lowe. To explain FLANN, we will continue with our puzzle solving example. Visualize a giant puzzle with hundreds of pieces scattered around. Your goal is to organize these pieces based on how well they fit together. Instead of randomly trying to match pieces,
FLANN uses some clever tricks to quickly figure out which pieces are most likely to go together. Instead of trying every piece against every other piece, FLANN streamlines the process by finding pieces that are approximately similar. This means it can make educated guesses about which pieces might fit well together, even if they're not an exact match. Under the hood, FLANN is uses something called k-D trees. Think of it as organizing the puzzle pieces in a special way. Instead of checking every piece against every other piece, FLANN arranges them in a tree-like structure that makes finding matches faster. In each node of the k-D tree, FLANN puts pieces with similar features together. It's like sorting puzzle pieces with similar shapes or colors into piles. This way, when you're looking for a match, you can quickly check the pile that's most likely to have similar pieces. Let's say you're looking for a "sky" piece. Instead of searching through all the pieces, FLANN guides you to the right spot in the k-D tree where the sky-colored pieces are sorted. FLANN also adjusts its strategy based on the features of the puzzle pieces. If you have a puzzle with lots of colors, it will focus on color features. Alternately, if it's a puzzle with intricate shapes, it pays attention to those shapes. By balancing speed and accuracy when finding matching features, FLANN substantially improves query time.

First, we create a dictionary to specify the algorithm we will use, for SIFT or SURF it looks like the following.

```python
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
```

For ORB, will use the parameters from the paper.

```python
FLANN_INDEX_LSH = 6
index_params = dict(
    algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2
)
```

We also create a dictionary to specify the maximum leafs to visit as follows.

```python
search_params = dict(checks=50)
```

Initiate SIFT detector.

```python
sift = cv2.SIFT_create()
```

Find the keypoints and descriptors with SIFT.

```python
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
```

We will now define the FLANN parameters. Here, trees is the number of bins you want.

```python
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)
```

We will only draw good matches, so create a mask.

```python
matchesMask = [[0, 0] for i in range(len(matches))]
```

We can perform a ratio test to determine good matches.

```python
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]
```

Now let's visualize the matches.

```python
draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    matchesMask=matchesMask,
    flags=cv2.DrawMatchesFlags_DEFAULT,
)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
```

![FLANN](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/feature-extraction-feature-matching/FLANN.png)

## Local Feature Matching with Transformers (LoFTR)

LoFTR was proposed in [LoFTR: Detector-Free Local Feature Matching with Transformers](https://arxiv.org/pdf/2104.00680.pdf) by Sun, et. al.
Instead of using feature detectors, LoFTR uses a learning-based approach to feature matching.

Let's keep it simple and use our puzzle example once again. Instead of simply comparing images pixel by pixel, LoFTR looks for specific key points, or features, in each image. It's like identifying the corners and edges of each puzzle piece. And just as someone really good a putting together a puzzle might focus on distinctive marks, LoFTR identifies these unique points in one image. These could be key landmarks or structures that stand out. As we have already learned, it is important that the matching algorithm handles changes in rotation or scale. If a feature is turned or resized, LoFTR would still recognize it. It's like solving puzzles where pieces may be flipped or adjusted. As LoFTR matches features, it assigns a similarity score to indicate how well the features align. Higher scores mean better matches. It's like giving a grade to how well one puzzle piece fits with another.
LoFTR is also invariant to certain transformations, meaning it can handle variations in lighting, angle, or perspective. This is crucial when dealing with images that might be photographed under different conditions. LoFTR's ability to robustly match features makes it valuable for tasks like image stitching, where you combine multiple images seamlessly by identifying and connecting common features.

We can use [Kornia](https://github.com/kornia/kornia) to find matching features in two images using LoFTR.

```bash
!pip install kornia  kornia-rs  kornia_moons opencv-python --upgrade
```

Import the necessary libraries.

```python
import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia_moons.viz import draw_LAF_matches
```

Load and resize the images.

```python
from kornia.feature import LoFTR

img1 = K.io.load_image(image1.jpg, K.io.ImageLoadType.RGB32)[None, ...]
img2 = K.io.load_image(image2.jpg, K.io.ImageLoadType.RGB32)[None, ...]

img1 = K.geometry.resize(img1, (512, 512), antialias=True)
img2 = K.geometry.resize(img2, (512, 512), antialias=True)
```

Indicate whether the image is an "indoor" or "outdoor" image.

```python
matcher = LoFTR(pretrained="outdoor")
```

LoFTR only works on grayscale images, so convert to images to grayscale.

```python
input_dict = {
    "image0": K.color.rgb_to_grayscale(img1),
    "image1": K.color.rgb_to_grayscale(img2),
}
```

Let's perform the inference.

```python
with torch.inference_mode():
    correspondences = matcher(input_dict)
```

Clean up the correspondences using Random Sample Consensus ([RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus)). This helps to deal with noise or outliers in the data.

```python
mkpts0 = correspondences["keypoints0"].cpu().numpy()
mkpts1 = correspondences["keypoints1"].cpu().numpy()
Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
inliers = inliers > 0
```

Finally, we can visualize the matches.

```python
draw_LAF_matches(
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts0).view(1, -1, 2),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1),
    ),
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts1).view(1, -1, 2),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1),
    ),
    torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
    K.tensor_to_image(img1),
    K.tensor_to_image(img2),
    inliers,
    draw_dict={
        "inlier_color": (0.1, 1, 0.1, 0.5),
        "tentative_color": None,
        "feature_color": (0.2, 0.2, 1, 0.5),
        "vertical": False,
    },
)
```

The best matches are visualized in green, while less certain matches are in blue.

![LoFTR](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/feature-extraction-feature-matching/LoFTR.png)

## Resources and Further Reading

- [FLANN Github](https://github.com/flann-lib/flann)
- [Image Matching Using SIFT, SURF, BRIEF and ORB: Performance Comparison for Distorted Images](https://arxiv.org/pdf/1710.02726.pdf)
- [ORB (Oriented FAST and Rotated BRIEF) tutorial](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html)
- [Kornia tutorial on Image Matching](https://kornia.github.io/tutorials/nbs/image_matching.html)
- [LoFTR Github](https://github.com/zju3dv/LoFTR)
- [OpenCV Github](https://github.com/opencv/opencv-python)
- [OpenCV Feature Matching Tutorial](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
- [OpenGlue: Open Source Graph Neural Net Based Pipeline for Image Matching](https://arxiv.org/abs/2204.08870)
