# Multimodal Based Video Models

As discussed in previous chapters, a video can be simply defined as a sequence of images. However, unlike simple images, videos contain various modalities such as sound, text, and movement. From this perspective, to properly understand a video, we must consider multiple modalities at the same time. In this chapter, we first briefly explain what modalities can exist in a video. Then, we introduce architectures that can learn by aligning videos with different modalities.

## What Modalities Are Present in Video?

Videos encompass a variety of modalities beyond just sequences of images. Understanding these different modalities is crucial for comprehensive video analysis and processing. The primary modalities present in videos include:

1. Visual Modality(Frames/Images): The most common modality, consisting of a sequence of images that provides the visual information for the video.
2. Audio Modality(Sound): Includes dialogue, background music, and environmental sounds that can convey contextual information about the video.
3. Text Modality(Captions/Subtitles): Appears as subtitles, captions, or on-screen text, offering explicit information related to the video’s context.
4. Motion Modality(Movement Dynamics): Captures temporal changes between video frames, reflecting movement and transitions.
5. Depth Modality: Represents the 3D spatial information of the video.
6. Sensor Modality: In some applications, videos may include modalities like temperature or biometric data.
    
<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Multimodal_Based_Video_Models/Modality_example.jpg" alt="Modality examples. The image is from the original LanuageBind paper">
</div>

Beyond the modalities mentioned above, videos can incorporate even more diverse types of modalities. Be sure to consider which modalities are necessary for your specific work or project. In the next section, we will explore video architectures that can align and represent these modalities jointly.

## Video and Text

### VideoBERT

**Overview**

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Multimodal_Based_Video_Models/Overview_VideoBERT.png" alt="VideoBERT model architecture.">
</div>

[VideoBERT](https://arxiv.org/abs/1904.01766) is an attempt to apply the BERT architecture directly to video data. Just like BERT in language models, the goal is to learn good visual-linguistic representation without any supervsion. For the text modality, VideoBERT uses ASR (Automatic Speech Recognition) to convert audio into text, and then obtains BERT token embeddings. For the video, it uses S3D to get token embeddings for each frame. 

**Key Features**

1. **Linguistic-visual alignment**: Classifies whether a given text and video frames are aligned or not.
2. **Masked Language Modeling**: Predicts masked tokens in the text (just like in BERT).
3. **Masked Frame Modeling**: Predicts the masked video frames (like MLM predicts masked tokens in text).

**Why It Matters**

VideoBERT was one of the first models to effectively integrate video-language understanding by learning joint representations.
Unlike previous methods, VideoBERT does not use a detection model for image-text labeling. Instead, it uses a *clustering algorithm* to enable Masked Frame modeling, allowing the model to predict masked frames without needing explicit labeled data.

### MERLOT

**Overview**

[MERLOT](https://arxiv.org/abs/2106.02636) is designed to improve multimodal reasoning by learning from large-scale video-text datasets. It focuses on understanding interactions between visual and textual information using no labeled data. By leveraging the large-scale unlabeled dataset **YT-Temporal-180M**,  **MERLOT** demonstrates strong performance in visual commonsense reasoning without relying on heavy visual supervision.
    
**Key Features**

1. Temporal Reordering Task (from [HERO](https://aclanthology.org/2020.emnlp-main.161.pdf))
2. Frame-Caption Matching Task (from [CBT](https://arxiv.org/pdf/1906.05743), [HAMMER](https://aclanthology.org/2020.emnlp-main.161.pdf))
3. Masked Language Modeling

Why It Matters

While the model architecture and training method are not entirely new, MERLOT achieves performance improvements by training on **YT-Temporal-180M**, a large-scale visual-text dataset. This extensive dataset enables the model to better understand temporal dynamics and multimodal interactions, leading to enhanced reasoning and prediction capabilities in video-language tasks.
    
<u>Note</u>: If you're looking to understand the detailed training process of MERLOT, make sure to refer to the MERLOT paper as well as earlier works like [HERO](https://aclanthology.org/2020.emnlp-main.161.pdf), [CBT](https://arxiv.org/pdf/1906.05743) and [HAMMER](https://aclanthology.org/2020.emnlp-main.161.pdf).

## Video and Audio, Text

### VATT(Visual-Audio-Text Transformer)

**Overview**

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Multimodal_Based_Video_Models/Overview_VATT.png" alt="VATT model architecture.">
</div>

[VATT](https://arxiv.org/abs/2104.11178) is a model designed for self-supervised learning from raw video, audio, and text. Different tokenization and positional encoding methods were applied to each modality, and VATT used the Transformer Encoder to effectively integrate the representations from the raw multimodal data. As a result, it achieved strong performance in various downstream tasks such as action recognition and text-to-video retrieval.

**Key Features**

1. Modality-Specific & Modality-Agnostic: The **modality-specific** version uses separate Transformer encoders for each modality, while the modality-agnostic version integrates all modalities with a single Transformer encoder. While modality-specific demonstrated better performance, the **modality-agnostic** still showed strong performance in downstream tasks with fewer parameters.
2. Droptoken: Due to the redundancies in video (with audio and text data), sampling only a subset of tokens allows for more efficient training.
3. Multimodal Contrastive Learning: Noise Contrastive Estimation (NCE) was used for video-audio pairs, while Multiple Instance Learning NCE (MIL-NCE) was applied to video-text pairs

**Why It Matter**
    
Previous models using transformers for video multimodal tasks tended to rely heavily on visual data and required extensive training time and computational complexity. In contrast, VATT utilizes **Droptoken** and **weight sharing** to learn powerful multimodal representations from raw visual, audio, and text data with relatively lower computational complexity.

### Video-Llama

**Overview**

[Video-LLaMA](https://arxiv.org/abs/2306.02858) is a multimodal framework designed to extend Large Language Models (LLMs) to understand both visual and auditory content in videos. It integrates video, audio and text, allowing the model to process and generate meaningful responses grounded in audiovisual information. Video-LLaMA addresses two key challenges: capturing temporal changes in visual scenes and integrating audio-visual signals into a unified system.
    
**Key Features**
    
Video-LLaMA has two branches

1. <u>Vision-Language branch</u> for processing video frames 
2. <u>Audio-Language branch</u> for handling audio signals. 

These branches are trained separately, undergoing both pre-training and fine-tuning phases. In the pre-training phase, the model learns to integrate different modalities, while in the fine-tuning phase, it focuses on improving its ability to follow instructions accurately.

In the case of the vision-language branch, there is an abundance of visual-text data available. However, for the audio-language branch, there is a lack of sufficient audio-text data. To address this, the model utilizes **ImageBind**, allowing the audio-language branch to be trained using visual-text data instead.
    
**Why It Matters**
    
Previous models struggled to handle both visual and auditory content together. Video-LLaMA addresses this by integrating these modalities in a single framework, capturing temporal changes in video and aligning audio-visual signals. It overcomes the limitations of earlier research by using cross-modal pre-training and instruction fine-tuning, achieving strong performance in multimodal tasks like video-based conversations without relying on separate models.

## Video and Multiple Modalities

### ImageBind

**Overview**

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Multimodal_Based_Video_Models/Overview_ImageBind.png" alt="ImageBind model architecture.">
</div>

ImageBind utilizes paired data between images and other modalities to integrate diverse modality representations, centering around image data.

**Key Features**

ImageBind unifies many kinds of modalities by utilizing pairs of images and other modalities. By leveraging *InfoNCE* as the loss function, the model aligns representations between the various inputs. Even in cases where paired data between non-image modalities are absent, ImageBind can effectively perform cross-modal retrieval and zero-shot tasks.
Additionally, the training process of ImageBind is relatively simple compared to other models and can be implemented in various ways.

**Why It Matters**

ImageBind's key contribution is its ability to integrate various modalities without the need for specific modality-paired datasets. Using images as a reference, it aligns and combines up to six different modalities — such as audio, text, depth, and more — into a unified representation space. The significance lies in its capacity to achieve this alignment across multiple modalities simultaneously, without requiring direct pairing for each combination, making it highly efficient for multimodal learning.

## Conclusion

We have briefly examined the different modalities present in videos and then explored models that integrate visual information with various other modalities. 
As time goes on, there is a growing body of research focused on integrating a wide range of modalities all at once. 

I'm excited to see what future models will emerge, integrating even more diverse modalities within the video content. The potential for advancing multimodal representation learning through videos feels limitless!
