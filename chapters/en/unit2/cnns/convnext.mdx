# ConvNext - A ConvNet for the 2020s (2022)

## Introduction
Recently, the breakthrough of Vision Transformers (ViTs) quickly superseded pure CNN models as the new state-of-the-art for image recognition.
Intriguingly, research has found that CNNs could adopt a significant portion of the design choices in Vision Transformers.
ConvNext represents a significant improvement to pure convolution models by incorporating techniques inspired by ViTs and achieving results comparable to ViTs in accuracy and scalability.


## Key Improvements
The author of the ConvNeXT paper starts building the model with a regular ResNet (ResNet-50), then modernizes and improves the architecture step-by-step to imitate the hierarchical structure of Vision Transformers.
The key improvements are:
- Training techniques
- Macro design
- ResNeXt-ify
- Inverted Bottleneck
- Large Kernel Sizes
- Micro Design

We will go through each of the key improvements.
These designs are not novel in itself. However, you can learn how researchers adapt and modify designs systematically to improve existing models.
To show the effectiveness of each improvement, we will compare the model's accuracy before and after the modification on ImageNet-1K.

![Block Comparison](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/block_comparison.png)


## Training Techniques
The researchers first discerned that, while architectural design choices are crucial, the quality of the training procedure also plays a pivotal role in influencing performance outcomes.
Inspired by DeiT and Swin Transformers, ConvNext closely adapts their training techniques. Some of the notable changes are:
- Epochs: Extending the epochs from the original 90 epochs to 300 epochs.
- Optimizer: Using AdamW optimizer instead of Adam optimizer, which differs in how it handles weight decay.
- Mixup (generates a weighted combination of random image pairs), Cutmix (cuts part of an image and replace it with a patch from another image), RandAugment (applies a series of random augmentations such as rotation, translation, and shear), and Random Erasing (randomly selects a rectangle region in an image and erases its pixels with random values) to increase training data.
- Regularization: Using Stochastic Depth and Label Smoothing as regularization techniques.

Modifying these training procedures has improved ResNet-50's accuracy from 76.1% to 78.8%.


## Macro Design
Macro design refers to the high-level structural decisions and considerations made in a system or model, such as the arrangement of layers, the distribution of computational load across different stages, and the overall structure.
Examining Swin Transformers' macro network, the authors have identified two noteworthy design considerations that benefit the performance of ConvNext.

### The stage compute ratio
The stage compute ratio refers to the distribution of computational load among the stages of a neural network model.
ResNet-50 has four main stages with (3, 4, 6, 3) blocks, which means it has a compute ratio of 3:4:6:3.
To follow Swin Transformer's compute ratio of 1:1:3:1, the researchers adjusted the number of blocks on each stage of the ResNet from (3, 4, 6, 3) to (3, 3, 9, 3) instead. 
Changing the stage compute ratio improves the model accuracy from 78.8% to 79.4%.

### Changing stem to Patchify
Typically, at the start of ResNet's architecture, the input is fed to a stem 7×7 convolution layer with stride 2, followed by a max pool, used to downsample the image by a factor of 4.
However, the authors discovered that substituting the stem with a convolutional layer featuring a 4×4 kernel size and a stride of 4 is more effective, effectively convolving them through non-overlapping 4x4 patches. 
Patchify serves the same purpose of downsampling the image by a factor of 4 while reducing the number of layers.
This Patchifying step slightly improves the model accuracy from 79.4% to 79.5%.


## ResNeXt-ify
ConvNext also adopts the idea of ResNeXt, explained in the previous section. 
ResNeXt demonstrates an improved trade-off between the number of floating-point operations (FLOPs) and accuracy compared to a standard ResNet.
By using depthwise convolution and 1 × 1 convolutions, we would have the separation of spatial and channel mixing - a characteristic also found in vision Transformers.
Using depthwise convolutions reduces the number of FLOPS and the accuracy.
However, by increasing the channels from 64 to 96, the accuracy is higher than the original ResNet-50 while maintaining a similar number of FLOPs.
This modification improves the model accuracy from 79.5% to 80.5%.


## Inverted Bottleneck
One common idea in every Transformer block is the usage of an inverted bottleneck, where the hidden layers are much bigger than the input dimension. 
This idea has also been used and popularized in Computer Vision by MobileNetV2.
ConvNext adopts this idea, having input layers with 96 channels and increasing the hidden layers to 384 channels.
By using this technique, it improves the model accuracy from 80.5% to 80.6%.

![Inverted Bottleneck Comparison](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/inverted_bottleneck.png)


## Large Kernel Sizes
A key factor contributing to the exceptional performance of Vision Transformer is its non-local self-attention, allowing for a broader receptive field of image features.
In Swin Transformers, the attention block window size is set to at least 7×7, surpassing the 3x3 kernel size of ResNext. 
However, before adjusting the kernel size, it is necessary to reposition the depthwise convolution layer, as shown in the image below. 
This repositioning enables the 1x1 layers to efficiently handle computational tasks, while the depthwise convolution layer functions as a more non-local receptor.
With this, the network can harness the advantages of incorporating bigger kernel-sized convolutions. 
Implementing a 7x7 kernel size maintains the accuracy at 80.6% but reduces the overall FLOPs efficiency of the model.

![Moving up the Depth Conv Layer](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/depthwise_moveup.png)


## Micro Design
In addition to the modifications stated, the author adds some micro-design changes to the model.
Mico design refers to low-level structural decisions such as the choice of activation functions and layer details.
Some of the notable micro changes are:
- Activation: Replacing ReLU activation with GELU (Gaussian Error Linear Unit) and eliminating all GELU layers from the residual block except for one between two 1×1 layers.
- Normalization: Fewer normalization layers by removing two BatchNorm layers and substituting BatchNorm with LayerNorm, leaving only one LayerNorm layer before the conv 1 × 1 layers.
- Downsampling Layer: Add a separate downsample layer in between ResNet stages
These final modifications improve the ConvNext accuracy from 80.6% to 82.0%.
The final ConvNext model exceeds Swin Transformer's accuracy of 81.3%.


## Model Code
You can go to [this HuggingFace documentation](https://huggingface.co/docs/transformers/model_doc/convnext) to learn how to integrate the ConvNext pipeline into your code.


## References
The paper "A ConvNet for the 2020s" proposed the ConvNext architecture in 2022 by a team of researchers from Facebook AI Research, including Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie.
The paper can be found [here](https://arxiv.org/abs/2201.03545) and the GitHub repo can be found [here](https://github.com/facebookresearch/ConvNeXt).