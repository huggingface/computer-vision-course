# Variational Autoencoders

## Introduction to Autoencoders
Autoencoders are a class of neural networks primarily used for unsupervised learning and dimensionality reduction. The fundamental idea behind autoencoders is to encode input data into a lower-dimensional representation and then decode it back to the original data, aiming to minimize the reconstruction error. The basic architecture of an autoencoder consists of two main components - `the encoder` and `the decoder`.
* **Encoder:** The encoder is responsible for transforming the input data into a compressed or latent representation. It typically consists of one or more layers of neurons that progressively reduce the dimensions of the input.
* **Decoder:** The decoder, on the other hand, takes the compressed representation produced by the encoder and attempts to reconstruct the original input data. Like the encoder, it often consists of one or more layers, but in the reverse order, gradually increasing the dimensions.

![Vanilla Autoencoder Image - Lilian Weng Blog](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/generative_models/autoencoder.png)

This encoder model consists of an encoder network (represented as $g_\phi$) and a decoder network (represented as $f_\theta$). The low-dimensional representation is learned in the bottleneck layer as $z$, and the reconstructed output is represented as $x' = f_\theta(g_\phi(x))$ with the goal of $x \approx x'$.

A common loss function used in such vanilla autoencoders is 

$$L(\theta, \phi) = \frac{1}{n}\sum_{i=1}^n (\mathbf{x}^{(i)} - f_\theta(g_\phi(\mathbf{x}^{(i)})))^2$$ 

which tries to minimize the error between the original image and the reconstructed one. This is also known as the `reconstruction loss`.


Autoencoders are useful for tasks such as data denoising, feature learning, and compression. However, traditional autoencoders lack the probabilistic nature that makes VAEs particularly intriguing and also useful for generational tasks.

## Variational Autoencoders (VAEs) Overview
Variational Autoencoders (VAEs) address some of the limitations of traditional autoencoders by introducing a `probabilistic approach` to encoding and decoding. The motivation behind VAEs lies in their ability to generate new data samples by sampling from a learned distribution in the latent space rather than from a latent vector as was the case with Vanilla Autoencoders which makes them suitable for generation tasks.
* **Probabilistic Nature:** Unlike deterministic autoencoders, VAEs model the latent space as a probability distribution. This produces a probability distribution function over the input encodings instead of just a single fixed vector. This allows for a more nuanced representation of uncertainty in the data. The decoder then samples from this probability distribution.
* **Role of Latent Space:** The latent space in VAEs serves as a continuous, structured representation of the input data. Since it is continuous by design, this allows for easy interpolation. Each point in the latent space corresponds to a potential output, enabling smooth transitions between different data points and also making sure that points which are closer to the latent space lead to similar generation

The concept can be elucidated through a straightforward example, as presented below. Encoders within a neural network are tasked with acquiring a representation of input images in the form of a vector. This vector encapsulates various features such as a subject's smile, hair color, gender, age, etc., denoted as a vector akin to [0.4, 0.03, 0.032, ...]. In this illustration, the focus is narrowed to a singular latent representation, specifically the attribute of a "smile."
![Autoencoders vs VAEs - Sciforce Medium](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/generative_models/comparison.png)
In the context of Vanilla Autoencoders (AE), the smile feature is encapsulated as a fixed, deterministic value. In contrast, Variational Autoencoders (VAEs) are deliberately crafted to encapsulate this feature as a probabilistic distribution. This design choice facilitates the introduction of variability in generated images by enabling the sampling of values from the specified probability distribution.

## Mathematics Behind VAEs
Understanding the mathematical concepts behind VAEs involves grasping the principles of probabilistic modeling and variational inference.
![Variational Autoencoder - Lilian Weng Blog](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/generative_models/vae.png)
* **Probabilistic Modeling:** In VAEs, the latent space is modeled as a probability distribution, often assumed to be a multivariate Gaussian. This distribution is parameterized by the mean and standard deviation vectors, which are outputs of the probabilistic encoder  \\( q_\phi(z|x) \\). This comprosises of our learned representation z which is further used to sample from the decoder as \\(p_\theta(x|z) \\).
* **Loss Function:** The loss function for VAEs comprises two components: the reconstruction loss (measuring how well the model reconstructs the input) similar to the vanilla autoencoders and the KL divergence (measuring how closely the learned distribution resembles a chosen prior distribution, usually gaussian). The combination of these components encourages the model to learn a latent representation that captures both the data distribution and the specified prior.
* **Encouraging Meaningful Latent Representations:** By incorporating the KL divergence term into the loss function, VAEs are encouraged to learn a latent space where similar data points are closer, ensuring a meaningful and structured representation. The autoencoder's loss function aims to minimize both the reconstruction loss and the latent loss. A smaller latent loss implies a limited encoding of information that would otherwise enhance the reconstruction loss. Consequently, the Variational Autoencoder (VAE) finds itself in a delicate balance between the latent loss and the reconstruction loss. This equilibrium becomes pivotal, as a `smaller latent loss` tends to result in generated images closely resembling those present in the training set but lacking in visual quality. Conversely, a `smaller reconstruction loss` leads to well-reconstructed images during training but hampers the generation of novel images that deviate significantly from the training set. Striking a harmonious balance between these two aspects becomes imperative to achieve desirable outcomes in both image reconstruction and generation.

In summary, VAEs go beyond mere data reconstruction; they generate new samples and provide a probabilistic framework for understanding latent representations. The inclusion of probabilistic elements in the model's architecture sets VAEs apart from traditional autoencoders. Compared to traditional autoencoders, VAEs provide a richer understanding of the data distribution, making them particularly powerful for generative tasks.

## References
1. [Lilian Weng's Awesome Blog on Autoencoders](https://lilianweng.github.io/posts/2018-08-12-vae/)
2. [Generative models under a microscope: Comparing VAEs, GANs, and Flow-Based Models](https://medium.com/sciforce/generative-models-under-a-microscope-comparing-vaes-gans-and-flow-based-models-344f20085d83)
3. [Autoencoders, Variational Autoencoders (VAE) and β-VAE](https://medium.com/@rushikesh.shende/autoencoders-variational-autoencoders-vae-and-%CE%B2-vae-ceba9998773d)
