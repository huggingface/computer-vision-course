# Introduction

In the last unit we have learned about multimodality and especially about how to fuse vision and language models to harness the best of the two worlds and outperform simple vision models in tasks like Zero-Shot Image Classification.
Another area where multimodal models have had an significant impact, are generative vision models. In this unit, we will have a deeper look at these types of Neural Networks.

## Definition

What are generative vision models and how do they differ from other models?

Mathematical models can generally be separated into two large families, generative models and discriminative models.
The main difference between discriminative models and generative models is that discriminative models learn boundaries that separate different classes, while generative models learn the distribution of different classes.

Discriminative models can be applied to standard computer vision tasks such as classification and regression,
these tasks can be expanded into more complex processes such as semantic segmentation or object detection.

For the sake of brevity, in this chapter, we will consider generative models that solve these tasks:

* noise to image (DCGAN)
* text to image (diffusion models)
* image to image (StyleGAN, cycleGAN, diffusion models)

This section will cover 2 kinds of generative models. GAN-based models, and diffusion-based models.

## Evaluation of generative models in computer vision

Generally, it is really hard to come up with meaningful metrics for evaluating generative models. Because often you don't have a solid "ground truth", and it is difficult to quantify the quality of an image. FID is the most commonly used metric, but it is not perfect.

Let's quickly go over FID. FID stands for Fréchet Inception Distance, it is an improvement on the Inception Score and was introduced in [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/pdf/1706.08500.pdf). FID is considered to be resistant to noise and certain artifacts that can be present in generated images. The lower the FID, the better.

It is calculated by constructing 2 distributions from the Inception-v3 features. The first distribution is calculated from the training data features, and the second distribution is calculated from the generated image features. Then the Fréchet distance between these 2 distributions is calculated, and that is your FID score. The lower this score, the better the perceived quality of the generated images. Here is a [short explanation](https://www.youtube.com/watch?v=9zTwSzXxNDo&t=398s) on FID.

Some other metrics you might come across are SSIM, PSNR, IS(Inception Score), and the recently introduced CLIP Score.

* PSNR (peak signal-to-noise ratio) can be interpreted almost as mean-squared-error. Generally, values from [25,34] are okay results while 34+ is very good.

* SSIM (Structural Similarity Index) is a metric in the range [0, 1] where 1 is a perfect match. The final index is calculated from 3 components: luminance, contrast, and structure. [this paper](https://arxiv.org/pdf/2006.13846.pdf) analyzes SSIM and its components if you're really interested.

* Inception score was introduced in [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf). It is calculated using the features on the inceptionv3 model. The higher the better. It is a mathematically very interesting metric, but has recently fallen out of favor.

* CLIP Score, this metric was introduced in [CLIPScore: A Reference-free Evaluation Metric for Image Captioning](https://arxiv.org/pdf/2104.08718.pdf) is used to evaluate the quality of text to image models. It is calculated by using the CLIP model to calculate the cosine similarity between the generated image and the text prompt. Its range is [0, 100], the higher the better.

 If you're *really curious* about FID. [The Role of ImageNet Classes in Fréchet Inception Distance](https://arxiv.org/pdf/2203.06026.pdf) tries to analyze what FID considers important in an image, and how the features pretrained on imagenet affect the FID score. It is a very interesting read.