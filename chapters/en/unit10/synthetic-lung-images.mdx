# Synthetic Data Generation Using DCGAN

We learned in Unit 5 that a GAN is a framework in machine learning where two neural networks, a Generator and a Discriminator, are in a constant duel. The Generator creates synthetic images, and the Discriminator tries to distinguish between real and fake images. They keep improving through this adversarial process, with the Generator getting better at creating realistic images, and the Discriminator getting better at distinguishing between fake and real images.

We now will look at how we can use a GAN to generate medical images, a domain that is challenged with small datasets, privacy concerns, and a limited amount of annotated samples. Researchers have used GANs to generate synthetic images such as lung X-ray images, retina images, brain scans, and liver images. In [GAN-based synthetic brain PET image generation](https://braininformatics.springeropen.com/counter/pdf/10.1186/s40708-020-00104-2.pdf), the authors created brain PET images for three different stages of Alzheimer’s disease. [GAN-based Synthetic Medical Image Augmentation for increased CNN Performance in Liver Lesion Classification](https://arxiv.org/abs/1803.01229) generated synthetic liver images. [BrainGAN: Brain MRI Image Generation and Classification Framework Using GAN Architectures and CNN Models](https://www.mdpi.com/1424-8220/22/11/4297) developed a framework for generating brain MRI images using multiple GAN architectures, and [A Novel COVID-19 Detection Model Based on DCGAN and Deep Transfer Learning](https://www.sciencedirect.com/science/article/pii/S1877050922007463) used DCGAN to generate synthetic lung X-ray images to aid in COVID-19 detection.

## DCGAN (Deep Convolutional Generative Adversarial Network)

DCGAN was proposed in [Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) by Radford, et al. and is the model many researchers have used to generate synthetic medical images. We are going to use it to generate synthetic lung images. Before we use DCGAN to train the model, we will briefly review its architecture. The generator network takes random noise as input and generates synthetic lung images, while the discriminator network tries to distinguish between real and synthetic images. It uses convolutional layers in both the generator and discriminator to capture spatial features effectively. DCGAN also replaces max-pooling with strided convolutions to downsample the spatial dimensions.

The generator has the following model architecture:

- The input is a vector a 100 random numbers and the output is a image of size 128*128*3.
- The model has 4 convolutional layers:
  - Conv2D layer
  - Batch Normalization layer
  - ReLU activation
- Conv2D layer with Tanh activation.

The discriminator has the following model architecture:

- The input is an image and the output is a probability indicating whether the image is fake or real.
- The model has one convolutional layer:
  - Conv2D layer
  - Leaky ReLU activation
- Three convolutional layers with:
  - Conv2D layer
  - Batch Normalization layer
  - Leaky ReLU activation
- Conv2D layer with Sigmoid.

**Data Collection**

First, we need to obtain a [dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2) of real lung images. We will be downloading the [Chest X-Ray Images (Pneumonia)](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia) dataset from the Hugging Face Hub.

Here is some information about the dataset:

    * From [Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867418301545%3Fshowall%3Dtrue):

    * The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

    * Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

    * For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

We will start by logging into the Hugging Face hub.

```python
from huggingface_hub import notebook_login

notebook_login()
```

Next, we will load the dataset.

```python
from datasets import load_dataset

dataset = load_dataset("hf-vision/chest-xray-pneumonia")
```

We will preprocess the lung images by resizing and normalizing the pixel values.

```python
import torchvision.transforms as transforms
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

transform = Compose(
    [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
)
```

During training, the generator aims to produce synthetic lung images that are indistinguishable from real images, while the discriminator learns to correctly classify the images as real or synthetic. We start by initializing the generator with random noise and will train for 100 epochs.

Let's visualize the progress:

![trainig-gif](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/dcgan_training_animation.gif)

## How did we do?

Here are 64 "good" synthetic images, defined as receiving a "real" label from the discriminator with a 70% probability.

![lung-images](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/good_images.png)

We can see that some of the synthetic lung images look good, but others are hazy. There are some important things to mention. First, researchers generating synthetic medical images typically employ a "human in the middle"--in this case, a expert radiologist--to evaluate the synthetic images. Then, only the images that fool the expert are included with the real data to train the model. Second, the generated images that appear to look OK (at least to an amateur) look very similar. This is another known issue with GANs - they can suffer from "mode collapse." Essentially, this happens when the generator starts producing the same type of output. Think of it as someone who has gotten a lot of praise for making chocolate chip cookies, and therefore, gets _really, really_ good at making those cookies but can't make any other type of cookies.

Given the known challenges associated with employing GANs to train high-quality medical images, some researchers have explored the use of diffusion models to generate lung images. Medfusion, a conditional latent DDPM for medical images, was proposed in [Diffusion Probabilistic Models Beat GAN On Medical 2D Images](https://arxiv.org/pdf/2212.07501.pdf). [Synthetically Enhanced: Unveiling Synthetic Data's Potential In Medical Imaging Research](https://arxiv.org/pdf/2311.09402.pdf), Khosravi et. al find that using a mix of real and synthetically generated lung images using a diffusion process improved model performance.

## Resources and Further Reading

- [A Novel COVID-19 Detection Model Based on DCGAN and Deep Transfer Learning](https://www.sciencedirect.com/science/article/pii/S1877050922007463)
- [Augmentation_Gan](https://github.com/rossettisimone/AUGMENTATION_GAN)
- [BrainGAN: Brain MRI Image Generation and Classification Framework Using GAN Architectures and CNN Models](https://www.mdpi.com/1424-8220/22/11/4297)
- [dentifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://www.cell.com/action/showPdf?pii=S0092-8674%2818%2930154-5)
- [Diffusion Probabilistic Models beat GANs on Medical Images](https://arxiv.org/abs/2212.07501)
- [DR-DCGAN: A Deep Convolutional Generative Adversarial Network (DC-GAN) for Diabetic Retinopathy Image Synthesis](<https://www.webology.org/data-cms/articles/20220204053948pmwebology%2019%20(2)%20-%2077%20.pdf>)
- [Deepfake Image Generation for Improved Brain Tumor Segmentation](https://aps.arxiv.org/abs/2307.14273)
- [GAN Lab](https://poloclub.github.io/ganlab/)
- [GANs for Medical Image Synthesis: An Empirical Study](https://arxiv.org/abs/2105.05318)
- [Medfusion Github repo](https://github.com/mueller-franzes/medfusion)
- [Medical image editing in the latent space of Generative Adversarial Networks](https://www.sciencedirect.com/science/article/pii/S2666521221000168?ref=pdf_download&fr=RR-2&rr=833e48fa5e777142)
- [Medical Image Synthesis with Context-Aware Generative Adversarial Networks](https://arxiv.org/abs/1612.05362)
- [MedSynAnalyzer](https://github.com/ayanglab/MedSynAnalyzer)[dcgan_faces_tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [pytorch-fid](https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py)
- [StudioGAN: A Taxonomy and Benchmark of GANs for Image Synthesis](https://arxiv.org/abs/2206.09479)
- [PyTorch-StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN)
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
