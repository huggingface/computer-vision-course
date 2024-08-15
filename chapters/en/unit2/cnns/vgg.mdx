
# Very Deep Convolutional Networks for Large Scale Image Recognition (2014)


## Introduction

The VGG architecture was developed in 2014 by Karen Simonyan and Andrew Zisserman from the Visual Geometry Group -and hence named VGG- at Oxford University. The model demonstrated significant improvements over the past models at that time- to be specific 2014 Imagenet challange also known as ILSVRC.



## VGG Network Architechture
  
- Inputs are 224x224 images. 
- Convolution kernel shape is (3,3) and max pooling window shape is (2,2).
- Number of channels for each convolutional layer 64 -> 128 -> 256 -> 512 -> 512. 
- VGG16 has 16 hidden layers (13 convolutional layers and 3 fully connected layers).
- VGG19 has 19 hidden layers (16 convolutional layers and 3 fully connected layers).

## Key Comparisons 

- VGG (16 or 19 layers) was relatively deeper than other SOTA networks at the time. AlexNet, the winning model for ILSVRC 2012 only has 8 layers.
- Multiple small (3X3) receptive field filters with ReLU activation instead of one large (7X7 or 11X11) filter lead to better learning of complex features. Smaller filters also mean fewer parameters per layer, with additional nonlinearity introduced in between.
- Multiscale training and inference. Each image was trained in multiple rounds with varying scales to ensure similar characteristics were captured at different sizes.  
- Consistency and simplicity of the VGG network make it easier to scale or modify for future improvements.


## PyTorch Example

Below you can find the PyTorch implementation of VGG19. 

```python
import torch.nn as nn


class VGG19(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG19, self).__init__()

        # Feature extraction layers: Convolutional and pooling layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                3, 64, kernel_size=3, padding=1
            ),  # 3 input channels, 64 output channels, 3x3 kernel, 1 padding
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2, stride=2
            ),  # Max pooling with 2x2 kernel and stride 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Pooling Layer
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(
                512 * 7 * 7, 4096
            ),  # 512 channels, 7x7 spatial dimensions after max pooling
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout layer with 0.5 dropout probability
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),  # Output layer with 'num_classes' output units
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # Pass input through the feature extractor layers
        x = self.avgpool(x)  # Pass Data through a pooling layer
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        x = self.classifier(x)  # Pass flattened output through the classifier layers
        return x
```

 