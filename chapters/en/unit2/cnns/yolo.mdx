# YOLO 

## A short introduction to Object Detection

Convolutional Neural Networks have taken a big step towards solving the image classification problem.
But there remains another big task to solve: object detection. Object detection not only requires categorizing the object from the image but also accurately predicting its location (in this case, the coordinates of the bounding boxes of the object) in the image. This is where the big breakthrough of YOLO came in. Before delving deeper into YOLO, let's go through the history of object detection algorithms using CNNs.


### RCNN, Fast RCNN, Faster RCNN

#### R- CNN (Region-based Convolutional Neural Networks)
RCNN is one of the simplest ways possible to use convolutional neural networks for object detection. In simple terms, the basic idea is to
detect a "region" and then use CNN to classify the region. So this is a multi-step process. Based on this idea the RCNN paper was shared in 2012[1]

The RCNN uses following steps, 

1. Use Selective Search algorithm to select a region. 
2. Use CNN based classifier to classify an object from the region. 

For training purpose, the paper proposed the following steps 

1. Make a dataset of regions detected from the Object detection dataset. 
2. Fine tune Alexnet model on the regions dataset. 
3. Then use the fine tuned model on the object detection dataset. 

The following is a basic pipeline of RCNN 
![rcnn](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/RCNN.png)


#### Fast RCNN 

Fast RCNN is focused on improvisation over the original RCNN. They added these four improvements

- Training in a single-stage instead of multi-stage like RCNN. Uses multi-task loss
- No disk storage required.
- Introduces ROI pooling layer to only get the features from the  Region of Interests.
- Trains an end-to-end model in contrary to multi-step RCNN / SPPnet models using multi-task loss.

![fast_rcnn](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Fast%20R-CNN.png)

#### Faster RCNN 

The Faster R-CNN completely removes the need for the Selective Search algorithm! These features resulted in an improvement of the inference time by 90% compared to that of Fast R-CNN!!
- It introduces RPN, Regional Proposal Network. The RPN is an attention based model which trains the model to give "attention" to the region of the image containing the object. 
- It merges RPN with Fast RCNN, making it an End-to-End object detection model. 

![Faster RCNN](https://cdn-uploads.huggingface.co/production/uploads/6141a88b3a0ec78603c9e784/n8eDqnlEvDS5SIKGoSUpz.png)

#### Feature Pyramid Network (FPN)
- A Feature Pyramid network is a kind of Inception model for Object detection.
- It first downscales the image into lower dimensional embeddings.
- Then it upscalses them again. 
- From every upscaled images it tries to predict the output (in this case the categories.)
- But there are also skip connections between the similar dimensional features as well! 

Please refer to the following images which are taken from the paper.[20]

   [FPN](https://huggingface.co/datasets/hf-vision/course-assets/blob/main/FPN.png)
   [FPN2](https://huggingface.co/datasets/hf-vision/course-assets/blob/main/FPN_2.png)

## YOLO architecture

YOLO was a ground breaking innovation of its time. A real time object detector, end to end trainable with a single network. 

### Before YOLO

The detection systems before consisted of utilizing image classifiers on patches of images. Systems like deformable parts models (DPM) used a sliding window approach where the classifier is run at evenly spaced locations over the entire image.

Other works like RCNN used a two-step detection. First, they detected many possible regions of interest, generated as bounding boxes by a Region Proposal Network. Then a classifier is run through all proposed regions to make final predictions. Post-processing, such as refining the bounding boxes, eliminating duplicate detections, and re-scoring the boxes based on other objects in the scene, needs to be done.

These complex pipelines are slow and hard to optimise because each individual component must be trained separately.

### YOLO
YOLO is a single step detector where the bounding box and the class of the object is predicted in the same pass, simultaneously. This makes the system super fast - 45-frames-per-second fast.

#### Reframing Object Detection
YOLO reframes the Object Detection task as a single regression problem, which predicts bounding box coordinates and class probabilities. 

In this design, we divide the image into an $S \times S$ grid. If the center of the object falls into a grid cell, that grid cell is responsible to detect that object. We can define $B$ as the maximum number of objects to be detected in each cell. So each grid cell predicts $B$ bounding boxes including confidence scores for each box. 

#### Confidence
The confidence score of a bounding box should reflect how accurately the box was predicted. It should be close to the IOU (Intersection over Union) of the ground truth box versus the predicted box. If the grid was not supposed to predict a box, then it should be zero. So this should encode the probability of the center of the box being present in the grid and the correctness of the bounding box.

Formally, 

$$\text{confidence} := P(\text{Object}) \times \text{IOU}_{\text{pred}}^{\text{truth}}$$

#### Coordinates
The coordinates of a bounding box are encoded in 4 numbers $(x, y, w, h)$. The $(x, y)$ coordinates represent the center of the box relative to the bounds of the grid cell. The width and height are normalized to image dimensions.

#### Class
The class probabilities is a $C$ long vecto,r representing conditional class probabilities of each class given an object existed in a cell. Each grid cell only predicts one vector, i.e a single class will be assigned to each grid cell and so all the $B$ bounding boxes predicted by that grid cell will have the same class.

Formally,
$$C_i = P(\text{class}_i \mid \text{Object})$$

At test time, we multiply the conditional class probabilities and the individual box confidence predictions, which gives us class-specific confidence scores for each box. These scores encode both the probability of that class appearing in the box and how well the predicted box fits the object.

$$\begin{align}
C_i \times \text{confidence} &= P(\text{class}_i \mid \text{Object}) \times P(\text{Object}) \times \text{IOU}_{\text{pred}}^{\text{truth}}\\
&=P(\text{class}_i) \times \text{IOU}_{\text{pred}}^{\text{truth}}
\end{align}
$$

To recap, we have an image, divided into $S \times S$ grid. Each grid cell contains $B$ bounding boxes consisting of 5 values - confidence + 4 coordinates and $C$ long vector containing conditional probabilities of each class. So, each grid cell is a $B \times 5 + C$ long vector. The whole grid is $S \times S \times (B \times 5 + C)$.

So if we have a learnable system which converts an image to an $S \times S \times (B \times 5 + C)$ feature map, we are one step closer to the task.

#### Network Architecture
In the original YOLOv1 design the input is an RGB image of size $448 \times 448$. The image is divided into a $S \times S = 7 \times 7$ grid, where each grid cell is responsible for detecting $B=2$ bounding boxes and $C=20$ classes. 

The network architecture is a simple convolutional neural network. The input image is passed through a series of convolutional layers, followed by a fully connected layer. The final layer outputs are reshaped to $7 \times 7 \times (2 \times 5 + 20) = 7 \times 7 \times 30$.

The YOLOv1 design took inspiration from GoogLeNet, which used 1x1 convolutions to reduce the depth of the feature maps. This was done to reduce the number of parameters and the amount of computation in the network. The network has 24 convolutional layers followed by 2 fully connected layers. It uses a linear activation function for the final layer, and all other layers use the leaky rectified linear activation:

$$\text{LeakyReLU}(x) = \begin{cases}
x & \text{if } x > 0\\
0.1x & \text{otherwise}
\end{cases}$$


See figure below for the architecture of YOLOv1.

![v1_arch](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/yolov1_arch.png)

#### Training
The network is trained end-to-end on the image and the ground truth bounding boxes. The loss function is a sum of squared error loss. The loss function is designed to penalize the network for incorrect predictions of bounding box coordinates, confidence and class probabilities. We will discuss the loss function in the next section.

YOLO predicts multiple bounding boxes per grid cell. At training time, we only want one bounding box predictor to be responsible for each object. We assign one predictor to be “responsible” for predicting an object based on which prediction has the highest current IOU with the ground truth. This leads to specialization between the bounding box predictors. Each predictor gets better at predicting certain sizes, aspect ratios, or classes of objects, improving overall recall. We will encode this information in the loss function for grid cell $i$ and bounding box $b$ using $\mathbb{1}{ib}^{\text{obj}}$. $\mathbb{1}{ib}^{\text{noobj}}$ is the opposite of $\mathbb{1}_{ib}^{\text{obj}}$.

##### Loss Function
Now that we have a learnable system which converts an image to a $S \times S \times (B\times5 + C)$ feature map, we need to train it.

A simple function to train such a system is to use a sum of squared error loss. We can use the squared error between the predicted values and the true values, i.e for bounding box coordinates, confidence and class probabilities. 

The loss for each grid cell $(i)$ can look like this:

$$
\mathcal{L}^{i} = \mathcal{L}^{i}_{\text{coord}} + \mathcal{L}^{i}_{\text{conf}} + \mathcal{L}^{i}_{\text{class}}\\
$$

$$\begin{align*}
\mathcal{L}^{i}_{\text{coord}} &= \sum_{b=0}^{B} \mathbb{1}_{ib}^{\text{obj}} \left[ \left( \hat{x}_{ib} - x_{ib} \right)^2 + \left( \hat{y}_{ib} - y_{ib} \right)^2 + 
\left( 
    \hat{w}_{ib} - w_{ib}
\right)^2 + 
\left( 
    \hat{h}_{ib} - h_{ib}
    \right)^2
\right]\\
\mathcal{L}^{i}_{\text{conf}} &= \sum_{b=0}^{B} (\hat{\text{conf}}_{i} - \text{conf}_{i})^2\\
\mathcal{L}^{i}_{\text{class}} &= \mathbb{1}_i^\text{obj}\sum_{c=0}^{C} (\hat{P}_{i} - P_{i})^2
\end{align*}
$$

where
- $\mathbb{1}_{ib}^{\text{obj}}$ is 1 if the $b$-th bounding box in the $i$-th grid cell is responsible for detecting the object, 0 otherwise.
- $\mathbb{1}_i^\text{obj}$ is 1 if the $i$-th grid cell contains an object, 0 otherwise.

But this loss function does not necessarily align well with the task of object detection. The simple addition of losses for both tasks (classification and localization) weights the loss equally. 

To rectify, YOLOv1 uses a weighted sum of squared error loss. First, we assign a separate weight to localization error called $\lambda_{\text{coord}}$. It is usually set to 5.

So the loss for each grid cell $(i)$ can look like this:
$$
\mathcal{L}^{i} = \lambda_{\text{coord}}
    \mathcal{L}^{i}_{\text{coord}} + \mathcal{L}^{i}_{\text{conf}} + \mathcal{L}^{i}_{\text{class}}\\
$$

In addition, many grid cells do not contain objects. The confidence is close to zero and thus the grid cells containing the objects often overpower the gradients. This makes the network unstable during training.

To rectify, we also weigh the loss from the confidence predictions in the grid cells that do not contain objects lower than in the grid cells that contain objects. We use a separate weight for the confidence loss called $\lambda_{\text{noobj}}$, which is usually set to 0.5.

So the confidence loss for each grid cell $(i)$ can look like this:
$$
\mathcal{L}^{i}_{\text{conf}} = \sum_{b=0}^{B} \left[
    \mathbb{1}_{ib}^{\text{obj}} \left( \hat{\text{conf}}_{i} - \text{conf}_{i} \right)^2 +
    \lambda_{\text{noobj}} \mathbb{1}_{ib}^{\text{noobj}} \left( \hat{\text{conf}}_{i} - \text{conf}_{i} \right)^2
\right]
$$

The sum of squared error for bounding box coordinates can be problematic. It equally weighs errors in large boxes and small boxes. The small deviations in large boxes should not be penalized as much as small deviations in small boxes.

To rectify, YOLOv1 uses a sum of squared error loss for the **square root** of the bounding box width and height. This makes the loss function scale invariant.

So the localization loss for each grid cell $(i)$ can look like this:

$$
\mathcal{L}^{i}_{\text{coord}} = \sum_{b=0}^{B} \mathbb{1}_{ib}^{\text{obj}} \left[ \left( \hat{x}_{ib} - x_{ib} \right)^2 + \left( \hat{y}_{ib} - y_{ib} \right)^2 +
\left(
    \sqrt{\hat{w}_{ib}} - \sqrt{w_{ib}}
\right)^2 +
\left(
    \sqrt{\hat{h}_{ib}} - \sqrt{h_{ib}}
\right)^2
\right]
$$

#### Inference
Inference is simple. We pass the image through the network and get the $S \times S \times (B \times 5 + C)$ feature map. We then filter out the boxes which have confidence scores less than a threshold. 

##### Non-Maximum Suppression
In rare cases, for large objects, the network tends to predict multiple boxes from multiple grid cells. To eliminate duplicate detections, we use a technique called Non-Maximum Suppression (NMS). NMS works by selecting the box with the highest confidence score and eliminating all other boxes with an IOU greater than a threshold. This is done iteratively until no overlapping boxes remain.

The end to end flow looks like this:
![nms](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/object-detection-gif.gif)

## Evolution of YOLO
So far, we have seen the basic characteristics of YOLO and how it allows for highly accurate and fast predictions. This was actually the first version of YOLO, called YOLOv1. YOLOv1 was released in 2015, and since then, multiple versions have been released. It was groundbreaking in terms of accuracy and speed, since it introduced the concept of using a single convolutional neural network (CNN) that processes the entire image at once, dividing it into an S × S grid. Each grid cell predicts bounding boxes and class probabilities directly. However, it suffered from poor localization and detection of multiple objects in small image areas. In the following years, many new versions were released by different teams to gradually improve the accuracy, speed, and robustness.

### **YOLOv2 (2016)** 
A year after releasing the first version, YOLOv2[5] came out. The improvements focused on both accuracy and speed, but also dealt with the localization problem. First, YOLOv2 replaced YOLOv1’s backbone architecture with Darknet-19, which is a variant of the Darknet architecture. Darknet-19 is lighter than the previous version’s backbone, and it consists of 19 convolutional layers followed by max-pooling layers. This led to YOLOv2 having the ability to capture more information. Also, it applied batch normalization to all convolutional layers and hence removed the dropout layers, dealing with the overfitting problem and increasing the mAP. It also introduced the concept of anchor boxes, which added prior knowledge to the width and height of the detected boxes (specifically, they used anchor boxes). Additionally, to deal with the poor localization, YOLOv2 predicted the class and objects for every anchor box and grid cell (now 13x13). So we have a maximum (for 5 anchor boxes) of 13x13x5 = 845 boxes.

### **YOLOv3 (2018)**
YOLOv3[6] again significantly improved the detection speed and accuracy by replacing the Darknet-19 architecture with the more complex but efficient Darknet-53. Also, it dealt with the localization problem better by using three different scales for object detection (13x13, 26x26, and 52x52 grids). This helped find objects of different sizes in the same area. It increased the bounding boxes to: 13 x 13 x 3 + 26 x 26 x 3 + 52 x 52 x 3 = 10,647. Non-Maximum Suppression (NMS) was still used to filter out redundant overlapping boxes.

### **YOLOv4 (2020)**
Back in 2020, YOLOv4[7] became one of the best detection models in terms of speed and accuracy, achieving state-of-the-art results on object detection benchmarks. The authors changed the backbone architecture again, opting for the faster and more accurate CSPDarknet53[8]. An important improvement of this version was the optimization for efficient resource utilization, making it suitable for deployment on various hardware platforms, including edge devices. Also, it included a number of augmentations before training that further improved the model's generalization. The authors included this improvement in a set of methodologies called bag-of-freebies. Bag-of-freebies are optimization methods that have a cost to the training process but aim to increase the model's accuracy in real-time detection without increasing the inference time.

### **YOLOv5 (2020)**
YOLOv5[9] translated the Darknet framework (written in C) to the more flexible and easy-to-use PyTorch framework. This version automated the previous anchor detection mechanism, introducing the auto-anchors. Auto-anchors train the model anchors automatically to match your data. During training, YOLO automatically uses k-means initially and genetic methods to evolve the new better-matched anchors and places them back into the YOLO model. Also, it offers different types of models that depend on the hardware constraints, with names similar to today’s YOLOv8 models: YOLOv5s, YOLOv5m, YOLOv5l, and YOLOv5x.

### **YOLOv6 (2022)**
The next version, YOLOv6[10][11], was released by the Meituan Vision AI Department under the article title: "YOLOv6: A Single-Stage Object Detection Framework for Industry." This team made further improvements in terms of speed and accuracy by focusing on five aspects:
1) Reparameterization using the RepVGG technique, which is a modified version of VGG with skip connections. During inference, these connections are fused to improve the speed.
2) Quantization of reparameterization-based detectors. Which added blocks that are called Rep-PANs.
3) Recognition of the importance of considering different hardware costs and capabilities for model deployment. Specifically, the authors tested the latency with lowpower GPUs (like Tesla T4) compared to the previous works which mostly used high-cost machines (like V100).
4) Introduction to new types of loss functions such as, Varifocal Loss for Classification. IoU Series Loss  for Bounding Box Regression, and Distribution Focal Loss.
5) Accuracy improvements during training using knowledge distillation.
In 2023, YOLOv6 v3[12] was released with the title "YOLOv6 v3.0: A Full-Scale Reload,", which introduced enhancements to the network architecture and training scheme, once again advancing speed and accuracy (evaluated on the COCO dataset) compared to previously released versions.

### **YOLOv7 (2022)**
YOLOv7 was released with the paper titled “YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors”[13][14] by the authors of YOLOv4. Specifically, this version of bag-of-freebies includes a new label assignment method called coarse-to-fine lead-guided label assignment and uses gradient flow propagation paths to analyze how re-parameterized convolution should be combined with different networks. They also proposed “extend” and “compound scaling” methods for the real-time object detector that can effectively utilize parameters and computation. Again, all these improvements took real-time object detection to a new state-of-the-art, outperforming previous releases.

### **YOLOv8 (2023)**
YOLOv8[15], developed by Ultralytics in 2023, became again the new SOTA. It introduced improvements on the backbone and neck alongside an anchor-free approach, which eliminates the need for predefined anchor boxes. Instead, predictions are made directly. This version supports a wide range of vision tasks, including classification, segmentation, and pose estimation. Additionally, YOLOv8 has scaling capabilities with pre-trained models available in various sizes: nano, small, medium, large, and extra-large, and can be easily fine-tuned on custom datasets.

### **YOLOv9 (2024)**
YOLOv9 was released with the paper titled “YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information”[16][17] by the same authors as YOLOv7 and YOLOv4. This paper highlights the issue of information loss that existing methods and architectures have during layer-by-layer feature extraction and spatial transformation. To address this issue, the authors proposed:

* Concept of programmable gradient information (PGI) to cope with the various changes required by deep networks to achieve multiple objectives.
* Generalized Efficient Layer Aggregation Network (GELAN), a new lightweight network architecture that achieves better parameter utilization than the current methods without sacrificing computational efficiency.

With these changes, YOLOv9 set new benchmarks on the MS COCO challenge.

Taking into consideration the timeline and the different licensing of the models we can create the following figure:
![yolo_evolution](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/yolo_evolution.png)

### **Note about the different versions**
This short chapter presented the history/evolution of YOLO in a linear way. However, this is not the case — many other versions of YOLO were released in parallel. Notice the release of YOLOv4 and YOLOv5 in the same year. Other versions that we did not cover include YOLOvX (2021), which was based on YOLOv3 (2018), and YOLOR (2021), which was based on YOLOv4 (2020), and many others.
Also, it is important to understand that the selection of the ‘best’ model version depends on the user requirements, such as speed, accuracy, hardware limitations, and user-friendliness. For example, YOLOv2 is very good at speed. YOLOv3 provides a balance between accuracy and speed. YOLOv4 has the best ability for adapting or being compatible across different hardware.

## Reference 
[1] [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524v5) <br />
[2] [Fast R-CNN](https://arxiv.org/abs/1504.08083) <br />
[3] [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) <br />
[4] [Feature Pyramid Network](https://arxiv.org/pdf/1612.03144.pdf) <br />
[5] [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) <br />
[6] [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) <br />
[7] [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934) <br />
[8] [YOLOv4 GitHub repo](https://github.com/AlexeyAB/darknet) <br />
[9] [Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/) <br />
[10] [YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications](https://arxiv.org/abs/2209.02976) <br />
[11] [YOLOv6 GitHub repo](https://github.com/meituan/YOLOv6) <br />
[12] [YOLOv6 v3.0: A Full-Scale Reloading](https://arxiv.org/abs/2301.05586) <br />
[13] [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696) <br />
[14] [YOLOv7 GitHub repo](https://github.com/WongKinYiu/yolov7) <br />
[15] [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) <br />
[16] [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616) <br />
[17] [YOLOv9 GitHub repo](https://github.com/WongKinYiu/yolov9) <br />
[18] [YOLOvX](https://yolovx.com/) <br />
[19] [You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/abs/2105.04206)
[20] [Feature Pyramid network Paper](https://arxiv.org/abs/1612.03144)
