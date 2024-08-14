# DEtection TRansformer (DETR)
## Overview of architecture
DETR is mainly used for object detection tasks, which is the process of detecting objects in an image. For example, the input of the model would be an image of a road, and the output of the model could be `[('car',X1,Y1,W1,H1),('pedestrian',X2,Y2,W2,H2)]`, in which X, Y, W, H represent the x, y coordinates denoting the location of the bounding box, as well as the width and height of the box. 
A traditional object detection model like YOLO consists of hand-crafted features like anchor box priors, which requires initial guesses of object locations and shapes, affecting downstream training. Post-processing steps are then used to remove overlapping bounding boxes, which require careful selection of its filtering heuristics. 
DEtection TRansformer, DETR for short, simplifies the detector by using an encoder-decoder transformer after the feature extraction backbone to directly predict bounding boxes in parallel, requiring minimal post-processing. 

![Architecture Diagram of DETR](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/DETR.png)
The model architecture of DETR begins with a CNN backbone, similar to other image-based networks, the output of which is processed and fed into a transformer encoder, resulting in N embeddings. The encoder embeddings are added to learned positional embeddings (called object queries) and used in a transformer decoder, generating another N embeddings. As a final step, each of the N embeddings are put through individual feed forward layers to predict the width, height, coordinates of the bounding box, as well as the object class (or whether there is an object). 

## Key Features

### Encoder-Decoder
As with other transformers, the transformer encoder expects the output of the CNN backbone to be a sequence. Thus, the feature map of size `[dimension, height, width]` is downsized and then flattened to `[dimension, less than height x width]`.
![Feature Maps of Encoder](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/DETR_FeatureMaps.png)
_**Left**: 12 out of 256 dimensions in the feature maps are visualized. Each dimension extracts some features of the original cat image while downsizing the original image. Some dimensions have a higher focus on the patterns on the cat; some have a higher focus on the bed sheets._ 
_**Right**: Keeping the original feature dimension of size 256, the width and height are further downsized and flattened into size 850._  
Since transformers are permutation invariant, positional embeddings are added to both the encoder and decoder to remind the model where on the image the embeddings come from. In the encoder, fixed positional encodings are used, while in the decoder, learned positional encodings (object queries) are used. Fixed encodings are similar to the ones used in the original Transformer paper, in which the encodings are defined by sinusoidal functions of varying frequencies at different feature dimensions. It gives the sense of position without having any learned parameters, indexed by the position on the image. Learned encodings are also indexed by the positions, but each position has a separate encoding that is learned throughout training to denote the positions in a way that the model understands. 

### Set-based Global Loss Function
In YOLO, a popular object detection model, the loss function comprises the bounding box, objectness (ie. the probability of an object existing in a region of interest), and class loss. The loss is calculated over multiple bounding boxes per each grid cell, which is a fixed number. On the other hand, in DETR, the architecture is expected to generate unique bounding boxes in a permutation-invariant manner (i.e., the order of the detections does not matter in the output, and the bounding boxes must vary and cannot all be the same). Thus, matching is required to assess how good the predictions are.

**Bipartite Matching**   
Bipartite matching is a way to compute one-on-one matching between the ground truth bounding boxes and the predicted boxes. It finds matches with the highest similarity between ground truth and predicted bounding boxes, as well as classes. This ensures the closest prediction would be matched with the corresponding ground truth in order to properly adjust the boxes and classes in the loss function. If matching is not done, predictions not aligned with the order of the ground truth would be marked as incorrect even if it were correct.


## Using DETR to Detect Objects
To see an example of how you can perform inference with DETR using Hugging Face transformers, see `DETR.ipynb`.

## Evolution of DETR
### Deformable DETR
Two of the main problems of DETR are a long and slow process of convergence and suboptimal small object detection. 
**Deformable Attention**   
The first problem is resolved by using deformable attention, which reduces the amount of sampling points to pay attention to. Traditional attention is inefficient due to global attention and heavily limit the resolution that the image can have. The model only attends to a fixed amount of sampling points around each reference point, and the reference points are learned by the model based on the input. For example, in an image of a dog, a reference point may be in the center of the dog, with sampling points near the ears, mouth, tail, etc. 

**Multi-scale Deformable Attention Module**   
The second problem is resolved similarly to YOLOv3, in which multi-scale feature maps are introduced. In convolutional neural networks, earlier layers extract smaller details (ex. lines) while later layers extract larger details (ex. wheels, ears). In a similar manner, different layers of deformable attention result in different levels of resolution. By connecting the outputs of some of these layers from the encoder to the decoder, it allows for the model to detect objects of a multitude of sizes. 

### Conditional DETR
Conditional DETR also sets out to resolve the problem of slow training convergence in the original DETR, resulting in convergence that is over 6.7 times faster. The authors found that the object queries are general and are not specific to the input image. Using **Conditional Cross-Attention** in the decoder, the queries can better localize the areas for bounding box regression.
![A decoder layer for Deformable DETR](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/DETR_DecoderLayer.png)
_Left: DETR Decoder Layer. Right: Deformable DETR Decoder Layer_   
The original DETR and Deformable DETR decoder layers are compared in the figure above, with the main difference being the query input of the cross-attention block. The authors make a distinction between content query c<sub>q</sub> (decoder self attention output) and spatial query p<sub>q</sub>. The original DETR simply adds them together. In Deformable DETR, they are concatenated, with c<sub>q</sub> focusing on the content of the object and p<sub>q</sub> focusing on the bounding box regions.   
The spatial query p<sub>q</sub> is the result of both the decoder embeddings and object queries projecting to the same space (to become T and p<sub>s</sub> respectively) and multiplied together. The previous layers' decoder embeddings contain information for the bounding box regions, and the object queries contains information of learned reference points for each bounding box. Thus, their projections combine into a representation that allows for cross-attention to measure their similarities with the encoder input and sinusoidal positional embedding. This is more effective than DETR which only uses object queries and fixed reference points.

## DETR Inference 

You can infer with existing DETR models on Hugging Face Hub like below:

```python
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# initialize the model
processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-101", revision="no_timm"
)
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-101", revision="no_timm"
)

# preprocess the inputs and infer
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# non max supression above 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.9
)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
```

Outputs are below.

```bash 
Detected cat with confidence 0.998 at location [344.06, 24.85, 640.34, 373.74]
Detected remote with confidence 0.997 at location [328.13, 75.93, 372.81, 187.66]
Detected remote with confidence 0.997 at location [39.34, 70.13, 175.56, 118.78]
Detected cat with confidence 0.998 at location [15.36, 51.75, 316.89, 471.16]
Detected couch with confidence 0.995 at location [-0.19, 0.71, 639.73, 474.17]
```

## PyTorch Implementation of DETR
The implementation of DETR from the original paper is shown below:
```python
import torch
from torch import nn
from torchvision.models import resnet50


class DETR(nn.Module):
    def __init__(
        self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers
    ):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers
        )
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = (
            torch.cat(
                [
                    self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                    self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )
        h = self.transformer(
            pos + h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1)
        )
        return self.linear_class(h), self.linear_bbox(h).sigmoid()
```
### Going line by line in the `forward` function: 
**Backbone**   
The input image is first put through a ResNet backbone and then a convolution layer, which reduces the dimension to the `hidden_dim`.
```python
x = self.backbone(inputs)
h = self.conv(x)
```
They are declared in the `__init__` function.
```python
self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
self.conv = nn.Conv2d(2048, hidden_dim, 1)
```

**Positional Embeddings**

While in the paper fixed and trained embeddings are used in the encoder and decoder respectively, the authors used trained embeddings for both in the implementation for simplicity.    
```python
pos = (
    torch.cat(
        [
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ],
        dim=-1,
    )
    .flatten(0, 1)
    .unsqueeze(1)
)
```
They are declared here as `nn.Parameter`. The row and column embeddings combined denote the locations in the image.
```python
self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
```
**Resize**   
Before going into the transformer, the features with size `(batch size, hidden_dim, H, W)` are reshaped to `(hidden_dim, batch size, H*W)`. This makes them a sequential input for the transformer.
```python
h.flatten(2).permute(2, 0, 1)
```
**Transformer**   
The `nn.Transformer` function takes in the first parameter as the input to the encoder, and the second parameter as the input of the encoder. As you can see, the encoder takes in the resized features added to the positional embeddings, while the decoder takes in `query_pos`, which is the decoder positional embedding.
```python
h = self.transformer(pos + h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1))
```
**Feed-Forward Network**   
In the end, the outputs, which is a tensor of size `(query_pos_dim, batch size, hidden_dim)`, is fed through two linear layers. 
```python
return self.linear_class(h), self.linear_bbox(h).sigmoid()
```
The first of which predicts the class. An additional class is added for the `No Object` class.
```python
self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
```
The second linear layer predicts the bounding box with an output size 4 for the xy coordinates, height and width.
```python
self.linear_bbox = nn.Linear(hidden_dim, 4)
```


## References
- [DETR](https://arxiv.org/abs/2005.12872) 
- [YOLO](https://arxiv.org/abs/1506.02640) 
- [YOLOv3](https://arxiv.org/abs/1804.02767) 
- [Conditional DETR](https://arxiv.org/abs/2108.06152) 
- [Deformable DETR](https://arxiv.org/abs/2010.04159) 
- [`facebook/detr-resnet-50`](https://huggingface.co/facebook/detr-resnet-50)
