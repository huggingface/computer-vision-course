# GoogLeNet

In this chapter we will go through a convolutional architecture called GoogleNet. 

## Overview

The Inception architecture, a convolutional neural network (CNN) designed for tasks in computer vision such as classification and detection, stands out due to its efficiency. It contains fewer than 7 million parameters and is significantly more compact than its predecessors, being 9 times smaller than AlexNet and 22 times smaller than VGG16. This architecture gained recognition in the ImageNet 2014 challenge, where Google's adaptation, named GoogLeNet (a tribute to LeNet), set new benchmarks in performance while utilizing fewer parameters compared to previous leading methods.


### Architectural Innovations

Before the advent of the Inception architecture, models like AlexNet and VGG demonstrated the benefits of deeper network structures. However, deeper networks typically entail more computational steps and can lead to issues such as overfitting and the vanishing gradient problem. The Inception architecture offers a solution, enabling the training of complex CNNs with a reduced count of floating-point parameters.

#### The Inception "Network In Network" Module 

In prior networks, such as AlexNet or VGG, the fundamental block is the convolution layer itself. However, Lin et al. 2013, introduced the concept of Network In Network, arguing that a single convolution is not necessarily a correct fundamental building block. It ought to be more complex. So, inspired by that, the Inception model authors decided to have a more complex building block called the Inception Module, aptly named after the famous movie - "The Inception" (dream in dream).

The Inception Module insists on applying convolution filters of different kernel sizes for feature extraction at multiple scales. For any input feature map, it applies a \\(1 \times 1\\) convolution, a 3x3 convolution, and a 5x5 convolution in parallel. In addition to convolution a max pooling operation is also applied. All four operations have padding and stride in such a way as to have the same spatial dimension. These features are concatenated and form the input to the next stage. See Figure 1.


![inception_naive](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/inception_naive.png)

Figure 1: Naive Inception Module

As we can see applying multiple convolutions at multiple scales with bigger kernel sizes, like 5x5, can increase the number of parameters drastically. This problem is pronounced as the input feature size (channel size) increases. So as we go deep in the network stacking these "Inception Modules", the computation will increase drastically. The simple solution is to reduce the number of features wherever computational requirements seem to increase. The major pain points of high computation are the convolution layers. The feature dimension is reduced by a computationally inexpensive \\(1 \times 1\\) convolution just before the 3x3 and 5x5 convolution. Let's see it with an example.

We want to convert a feature map of \\( S \times S \times 128 \\) to \\( S \times S \times 256 \\) via a 5x5 convolution. The number of parameters (excluding biases) is 5\*5\*128\*256 = 819,200. However, if we reduce the feature dimension first by a \\(1 \times 1\\) convolution to 64, then the number of parameters(excluding biases) is \\( 1\times 1\times 128\times 64 + 5\times 5\times 64\times 256 = 8,192 + 409,600 = 417,792 \\). That means the number of parameters was reduced by almost half!

We would also want to reduce the output features of max pooling before concatenating with the output feature map. So, we add one more \\( 1\times 1 \\) convolution after the max-pooling layer. We also add a ReLU activation after each \\( 1\times 1 \\) convolution increasing non-linearity and complexity of the module. See Figure 2.

![inception_reduced](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/inception_reduced.png)

Figure 2: Inception Module

Also, because of the parallel operations of convolutions at multiple scales, we are ensuring more operations without going deeper into the network, essentially mitigating the vanishing gradient problem.

#### Average Pooling

In prior networks, like AlexNet or VGG, the final layers would be a few fully connected layers. These fully connected layers, due to their large number of units would contribute to most of the parameters in a network. For example, 89% of the parameters of VGG16 are in the final three fully connected layers. 95% of parameters in AlexNet are in the final fully connected layers. This need can be attributed to the premise that a Convolutional layer is not necessarily complex enough. 


However, with an Inception block at our disposal, we do not need fully connected layers and a simple average pooling along the spatial dimensions should be enough. This was also derived from the Network in Network paper. However, GoogLeNet included one fully connected layer. They reported an increase of 0.6% in top-1 accuracy. 


GoogLeNet has only 15% of the parameters in the fully connected layers.


#### Auxiliary Classifiers

With the introduction of compute saving \\( 1 \times 1 \\) convolution and the replacement of multiple fully connected layers with average pooling, the parameters of this network are reduced significantly, which means we can add more layers and go deeper into the network. However, stacking layers can cause the problem of vanishing gradient, where the gradients get smaller and close to zero while propagating back to the initial layers of the network.

The paper introduces the auxiliary classifiers - branch out a few small classifiers from the layers in between and add the loss from these classifiers to the total loss(with less weightage). This ensures that the layers close to the input also receive gradients of decent magnitude.

The auxiliary classifier consists of 
- An average pooling layer with \\( 5 \times 5 \\) filter size and stride 3.
- A \\( 1 \times 1 \\) convolution with 128 filters for dimension reduction and rectified linear activation.
- A fully connected layer with 1024 units and rectified linear activation.
- A dropout layer with 70% ratio of dropped outputs.
- A linear layer with softmax loss as the classifier.

These auxiliary classifiers are removed at inference time. However, minimal gains are achieved from using auxiliary classifiers (0.5%).

![googlenet_aux_clf](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/googlenet_auxiliary_classifier.jpg)

Figure 3: An Auxiliary Classifier

### Architecture - GoogLeNet

The complete architecture of GoogLeNet is shown in Figure below. All convolutions, including inside the inception block, use ReLU activation. It starts with two convolution(s) and max-pooling blocks. This is followed by a block of two inception modules (3a and 3b) and a max pooling. This follows a block of 5 inception blocks (4a, 4b, 4c, 4d, 4e) and a max pooling after. The auxiliary classifiers are taken out from outputs of 4a and 4d. Two inception blocks follow (5a and 5b). After this, an average pooling and a fully connected layer of 128 units are used. 

![googlenet_arch](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/googlenet_architecture.png)
Figure 4: Complete GoogLeNet Architecture

### Code 


```python
import torch
import torch.nn as nn


class BaseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BaseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_proj):
        super(InceptionModule, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1),
            nn.ReLU(True),
        )

        self.b2 = nn.Sequential(
            BaseConv2d(in_channels, n3x3red, kernel_size=1),
            BaseConv2d(n3x3red, n3x3, kernel_size=3, padding=1),
        )

        self.b3 = nn.Sequential(
            BaseConv2d(in_channels, n5x5red, kernel_size=1),
            BaseConv2d(n5x5red, n5x5, kernel_size=5, padding=2),
        )

        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            BaseConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.7):
        super(AuxiliaryClassifier, self).__init__()
        self.pool = nn.AvgPool2d(5, stride=3)
        self.conv = BaseConv2d(in_channels, 128, kernel_size=1)
        self.relu = nn.ReLU(True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, use_aux=True):
        super(GoogLeNet, self).__init__()

        self.use_aux = use_aux
        ## block 1
        self.conv1 = BaseConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.lrn1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)

        ## block 2
        self.conv2 = BaseConv2d(64, 64, kernel_size=1)
        self.conv3 = BaseConv2d(64, 192, kernel_size=3, padding=1)
        self.lrn2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        ## block 3
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        ## block 4
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        ## block 5
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        ## auxiliary classifier
        if self.use_aux:
            self.aux1 = AuxiliaryClassifier(512, 1000)
            self.aux2 = AuxiliaryClassifier(528, 1000)

        ## block 6
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        ## block 1
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.lrn1(x)

        ## block 2
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.lrn2(x)
        x = self.maxpool2(x)

        ## block 3
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        ## block 4
        x = self.inception4a(x)
        if self.use_aux:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.use_aux:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        ## block 5
        x = self.inception5a(x)
        x = self.inception5b(x)

        ## block 6
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.use_aux:
            return x, aux1, aux2
        else:
            return x
```
