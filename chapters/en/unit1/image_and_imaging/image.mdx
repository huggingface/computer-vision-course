# Image

It might be weird that we will explain to you what an image is in a Computer Vision Course. Presumably, you got here in the first place because you wanted to know more about processing image and video formats. It seems trivial, but you are in here for a surprise! When it comes to images, there is much more than meets the eye (pun-intended).

## Definition of Image

An image is a visual representation of an object, a scene, a person, or even a concept. They can be photographs, paintings, drawings, schemes, scans and more! One of the more surprising things is that an image is also a function. More precisely, an image is an n-dimensional function. We will first consider it to be two-dimensional \\(n=2\\). We will call it \\(F(X,Y)\\), where \\(X\\) and \\(Y\\) are spatial coordinates. Do not get distracted by the fancy name. Spatial coordinates are just the system that we use to describe the positions of objects in a physical space with the most common one being the 2D Cartesian system. The amplitude of F at a pair of coordinates \\(x_i, y_i\\) is the intensity or gray level of the image at that point. The intensity is what gives you the perception of light and dark. Typically, when we have the coordinate pair \\(x_1\\) and \\(y_1\\), we refer to them as pixels (picture elements).

Images are discrete, and the processes involved in assembling them are continuous. The image generation processes will be discussed in the next chapter. Right now, what matters is that the value of \\(F\\) at specific coordinates holds a physical meaning. The function \\(F(X,Y)\\) is characterized by two components: the amount of illumination from the source and the amount of illumination reflected by the object in the scene. Intensity images are also constrained in their intensity since the function is typically non-negative, and their values are finite. 

That is not the only way one can create an image.  Sometimes, they are created by computers with or without the help of AI.  We have a dedicated chapter for images that do have a little helping hand from AI. Most of the terminology we will introduce here will still be applicable.

A different type of image is volumetric or 3D images. The number of dimensions in 3D images is equal to three. As a result, we have a \\(F(X,Y,Z)\\) function. Most of our reasoning still applies, with the only difference being that the triplet \\(x_i,y_i,z_i\\) is called a voxel (volume element). These images can be acquired in 3D; that is, the images are acquired in a way that is reconstructed in a 3D space. Examples of such images include medical scans, magnetic resonance, and certain types of microscopes. It is also possible to reconstruct a 3D image from a 2D one. Reconstructing is a challenging task, and it also has its dedicated chapter.

Now that we have discussed space, we can talk about color. The good news is you have likely heard of image channels, too. You might not understand what they mean, but fear not! Image channels are simply the different color components that make an image. In reference to the \\(F(X,Y)\\), we will have \\(F\\) for each color component. Each color has its own intensity level. For a channel that picks up the color red, a high intensity means that the color is very red and a low intensity means that there is little to no red in there.

If you're only looking at the \\(F(x,y)\\) for one color, it ranges from 0 to 255, where 0 represents no intensity and 255 represents maximum intensity. In a different color system, combining these values might differ. So, it is important to understand where your data comes from when interpretating these values.

There are special types of images where the coordinates \\(F(x_i,y_i)\\) do not describe an intensity value, but instead label a pixel. The simplest example of an operation that results in such an image is separating foreground and background. Everything that is foreground receives the label 1, and everything that is background receives the label 0. These images are commonly referred to as labeled images. When there are only two labels, like our example, we call them binary images or masks. 

You may have heard of 4D or even 5D images. This terminology is mostly used by people in the biomedical field and microscopists. Again, fear not! This naming came to be from people who image volumetric data in time, with different channels, or different imaging modalities (i.e. photo and an x-ray). The idea is that each new source of information becomes an extra dimension. Thus, a 5D image is a volumetric image (3D) imaged in time (4D) and using different channels (5D).

But how are images represented in computers? Most commonly by matrices. It is easy to picture an image as a 2D numerical array. This is an advantage because computers handle arrays really well. Seeing matrices as images helps to understand some of the processes in convolution neural networks and in image preprocessing. We will see more details latter on.

Alternatively, images can be represented as graphs where each node is a coordinate, and the edges are the neighboring coordinates. Take a moment to let that sink in. It also means that the algorithms and models used for graphs can also be used for images! The inverse can also be true - you can transform a graph into an image and analyze it as if it were a picture.

So far, we proposed a rather flexible definition of an image. This definition can accommodate different ways of acquiring visual data, but they all highlight the same crucial aspect: images are data points that contain a lot of spatial information. The key differences are the spatial resolution (either 2D or 3D), their color systems (RGB or others), and whether they have a time component attached to them.


## Images vs Other Data Types

### Difference between Image and Video

If you've been tuned in,  you may have caught on to the idea that videos are a visual representation of images with a time component attached. For 2D image acquisition, you can add a time dimension such that \\(F(X,Y,T)\\) becomes your imaging function. 

Images can naturally have a hidden component in time. They are, after all, taken at a specific point in time, and different images may be related in time, too. However, images and videos differ in how they sample this temporal information. An image is a static representation at a single point in time, while a video is a sequence of images played at a rate that creates an illusion of motion. This rate is what we can call frames per second. 

This is so fundamental, that this course has a dedicated chapter to video. There, we will go over the adaptations required to deal with this added dimension.

### Images vs Tabular Data

In tabular data, dimensionality is usually defined by the number of features (columns) describing one data point. In visual data, dimensionality usually refers to the number of dimensions that describe your data. For a 2D image, we usually refer to numbers \\(x_i\\) and \\(y_i\\) as the image size.  

 Another aspect is the generation of features that describe visual data. They are generated by traditional preprocessing or learned through deep learning methods. We refer to this as feature extraction. It involves different algorithms discussed in more detail in the feature extraction chapter. It contrasts with the feature engineering for tabular data, where new features are built upon existing ones.

Tabular data often require the handling of missing values, encoding categorical variables and re-scaling numerical features. The analogous process for image data is image resizing and intensity value normalization. We call these processes preprocessing and we will discuss them in greater detail in the chapter "preprocessing for computer vision".

### Key Differences

The table below summarizes the key aspects of different data types. 


|   | Feature                | Image                                                | Video                                                 | Audio                                                                     | Tabular Data                                                         |
|---|------------------------|------------------------------------------------------|-------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------|
| 1 | Type                   | Single moment in time                                | Sequence of images over time                          | Single moment in time                                                     | Structured data organized in rows and columns                        |
| 2 | Data Representation    | Typically a 2D array of pixels                       | Typically a 3D array of frames                        | Typically a 1D array of audio samples                                     | Typically a 2D array of features as columns and individual data sample as rows (i.e. spreadsheet, database tables                                         |
| 3 | File types             | JPEG,PNG,RAW, etc.                                   | MP4,AVI, MOV, etc.                                    | WAV, MP3, FLAC, etc.                                                      | CSV, Excel (.xlsx, .xls), Database formats, etc.                     |
| 4 | Data Augmentation	     | Flipping, rotating, cropping	                        | Temporal jittering, speed variations, occlusion       | Background noise addition, reverberation, spectral manipulation           |  ROSE, SMOTE, ADASYN                                |
| 5 | Feature Extraction	    | Edges, textures, colors	                             | Edges, textures, colors, optical flow, trajectories   | Spectrogram, Mel-Frequency Cepstral Coefficients (MFCCs), Chroma features | Statistical analysis, Feature engineering, Data aggregation          |
| 6 | Learning Models	       | CNNs                                                 | RNNs, 3D CNNs                                         | CNNs, RNNs                                                                | Linear Regression, Decision Trees, Random Forests, Gradient Boosting |
| 7 | Machine Learning Tasks | Image classification, Segmentation, Object Detection | Video action recognition, temporal modeling, tracking | Speech recognition, speaker identification, music genre classification    | Regression, Classification, Clustering                               |
| 8 | Computational Cost	    | Less expensive	                                      | More expensive                                        | Moderate to high                                                          | Generally less expensive compared to others                          |
| 9 | Applications           | Facial recognition for security access control	      | Sign language interpretation for live communication   | Voice assistants, Speech-to-text, Music genre classification              | Predictive modeling, Fraud detection, Weather forecasting            |




