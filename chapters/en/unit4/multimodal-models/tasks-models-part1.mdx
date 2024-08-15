# Multimodal Tasks and Models

In this section, we will briefly look at the different multimodal tasks involving Image and Text modalities and their corresponding models. Before diving in, let's have a small recap on what is meant by "multimodal" which was covered in previous sections. The human world is a symphony of diverse sensory inputs. We perceive and understand through sight, sound, touch, and more. This multimodality is what separates our rich understanding from the limitations of traditional, unimodal AI models. Drawing inspiration from human cognition, multimodal models aim to bridge this gap by integrating information from multiple sources, like text, images, audio, and even sensor data. This fusion of modalities leads to a more comprehensive and nuanced understanding of the world, unlocking a vast range of tasks and applications.

## Examples of Tasks
Before looking into specific models, it's crucial to understand the diverse range of tasks involving image and text. These tasks include but are not limited to:

- **Visual Question Answering (VQA) and Visual Reasoning:** Imagine a machine that looks at a picture and understands your questions about it. Visual Question Answering (VQA) is just that! It trains computers to extract meaning from images and answer questions like "Who's driving the car?" while Visual Reasoning is the secret sauce, enabling the machine to go beyond simple recognition and infer relationships, compare objects, and understand scene context to give accurate answers. It's like asking a detective to read the clues in a picture, only much faster and better!

- **Document Visual Question Answering (DocVQA):** Imagine a computer understanding both the text and layout of a document, like a map or contract, and then answering questions about it directly from the image. That's Document Visual Question Answering (DocVQA) in a nutshell. It combines computer vision for processing image elements and natural language processing to interpret text, allowing machines to "read" and answer questions about documents just like humans do. Think of it as supercharging document search with AI to unlock all the information trapped within those images.

- **Image captioning:** Image captioning bridges the gap between vision and language. It analyzes an image like a detective, extracting details, understanding the scene, and then crafting a sentence or two that tells the story – a sunset over a calm sea, a child laughing on a swing, or even a bustling city street. It's a fascinating blend of computer vision and language, letting computers describe the world around them, one picture at a time.

- **Image-Text Retrieval:** Image-text retrieval is like a matchmaker for images and their descriptions. Think of it like searching for a specific book in a library, but instead of browsing titles, you can use either the picture on the cover or a brief summary to find it. It's like a super-powered search engine that understands both pictures and words, opening doors for exciting applications like image search, automatic captioning, and even helping visually impaired people "see" through text descriptions.

- **Visual grounding:** Visual grounding is like connecting the dots between what we see and say. It's about understanding how language references specific parts of an image, allowing AI models to pinpoint objects or regions based on natural language descriptions. Imagine asking "Where's the red apple in the fruit bowl?" and the AI instantly highlights it in the picture - that's visual grounding at work!

- **Text-to-Image generation:** Imagine a magical paintbrush that interprets your words and brings them to life! Text-to-image generation is like that; it transforms your written descriptions into unique images. It's a blend of language understanding and image creation, where your text unlocks a visual world from photorealistic landscapes to dreamlike abstractions, all born from the power of your words.

## Visual Question Anwering (VQA) and Visual Reasoning
![VQA](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/multimodal_fusion_text_vision/vqa_visual_reasoning.png) *Example of Input (Image + Text) and Output (Text) for the VQA and Visual Reasoning Models [[1]](#pretraining-paper)*

**Visual Question Answering (VQA)**
- **Input:** An image-question pair (image and a question about it).
- **Output:** In multiple-choice setting: A label corresponding to the correct answer among pre-defined choices.
In open-ended setting: A free-form natural language answer based on the image and question.
- **Task:** Answer questions about images. (Most VQA models treat as a classification problem with pre-defined answers). See the above example for the reference.

**Visual Reasoning**
- **Input:** Varies depending on the specific visual reasoning task:
    - VQA-style tasks: Image-question pairs.
    - Matching tasks: Images and text statements.
    - Entailment tasks: Image and text pair (potentially with multiple statements).
    - Sub-question tasks: Image and a primary question with additional perception-related sub-questions.
- **Output:** Varies depending on the task:
    - VQA: Answers to questions about the image.
    - Matching: True/False for whether the text is true about the image(s).
    - Entailment: Prediction of whether the image semantically entails the text.
    - Sub-question: Answers to the sub-questions related to perception.
- **Task:** Performs various reasoning tasks on images. See the above example for the reference.

In general, both VQA and Visual Reasoning are treated as *Visual Question Answering (VQA)* task. Some of the popular models for VQA tasks are:
- **BLIP-VQA**: It is a large pre-trained model for visual question answering (VQA) tasks developed by Salesforce AI. It uses a "Bootstrapping Language-Image Pre-training" (BLIP) approach, which leverages both noisy web data and caption generation to achieve state-of-the-art performance on various vision-language tasks. You can use the BLIP in huggingface as follows:
```python
from PIL import Image
from transformers import pipeline

vqa_pipeline = pipeline(
    "visual-question-answering", model="Salesforce/blip-vqa-capfilt-large"
)

image = Image.open("elephant.jpeg")
question = "Is there an elephant?"

vqa_pipeline(image, question, top_k=1)
```
- **Deplot**: It is a one-shot visual language reasoning model trained on translating plots and charts to text summaries. This enables its integration with LLMs for answering complex questions about the data, even with novel human-written queries. DePlot achieves this by standardizing the plot-to-table translation task and leveraging Pix2Struct architecture, surpassing previous SOTA on chart QA with just a single example and LLM prompting. You can use the Deplot in huggingface as follows:
```python
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image

processor = Pix2StructProcessor.from_pretrained("google/deplot")
model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")

url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    images=image,
    text="Generate underlying data table of the figure below:",
    return_tensors="pt",
)
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))
```
- **VLIT**: It is a Vision-and-Language Transformer (ViLT) model, utilizing a transformer architecture without convolutions or region supervision, fine-tuned on the VQAv2 dataset for answering natural language questions about images. The base ViLT model boasts a large architecture (B32 size) and leverages joint image and text training, making it effective for various vision-language tasks, particularly VQA, achieving competitive performance. You can use VLIT in HuggingFace as follows:
```python
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

# prepare image + question
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "How many cats are there?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
```
Learn more about how to train and use VQA models in HuggingFace `transformers` library [here](https://huggingface.co/docs/transformers/v4.36.1/tasks/visual_question_answering).

## Document Visual Question Answering (DocVQA)
![DocVQA](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/multimodal_fusion_text_vision/doc_vqa.jpg)
*Example of Input (Image + Text) and Output (Text) for the Doc VQA Model. [[2]](#doc-vqa-paper)*
- **Input:**
    - Document image: A scanned or digital image of a document, containing text, layout, and visual elements.
    - Question about the document: A natural language question posed in text format.
- **Task:**
    - Analyze and understand: The DocVQA model must process both the visual and textual information within the document to fully comprehend its content.
    - Reason and infer: The model needs to establish relationships between visual elements, text, and the question to draw relevant conclusions.
    - Generate a natural language answer: The model must produce a clear, concise, and accurate answer to the question in natural language text format. See the above example for the reference.

- **Output:** Answer to the question: A text response that directly addresses the query and accurately reflects the information found in the document.

Now, let's look at the some of the popular DocVQA models in the HuggingFace:
- **LayoutLM:** It is a pre-trained neural network that understands document images by jointly analyzing both the text and its layout. Unlike traditional NLP models, it considers factors like font size, position, and proximity to learn relationships between words and their meaning in the context of the document. This allows it to excel at tasks like form understanding, receipt analysis, and document classification, making it a powerful tool for extracting information from scanned documents. You can use LayoutLM in HuggingFace as follows:
```python
from transformers import pipeline
from PIL import Image

pipe = pipeline("document-question-answering", model="impira/layoutlm-document-qa")

question = "What is the purchase amount?"
image = Image.open("your-document.png")

pipe(image=image, question=question)

## [{'answer': '20,000$'}]
```
- **Donut:** Also known as OCR-free Document Understanding Transformer, is a state-of-the-art image processing model that bypasses traditional optical character recognition (OCR) and directly analyzes document images to understand their content. It combines a vision encoder (Swin Transformer) with a text decoder (BART) to extract information and generate textual descriptions, excelling in tasks like document classification, form understanding, and visual question answering. Its unique strength lies in its "end-to-end" nature, avoiding potential errors introduced by separate OCR steps and achieving impressive accuracy with efficient processing. You can use Donut model in HuggingFace as follows:
```python
from transformers import pipeline
from PIL import Image

pipe = pipeline(
    "document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa"
)

question = "What is the purchase amount?"
image = Image.open("your-document.png")

pipe(image=image, question=question)

## [{'answer': '20,000$'}]
```
- **Nougat:** It is a visual transformer model, trained on millions of academic papers, that can directly "read" scanned PDFs and output their content in a structured markup language, even understanding complex elements like math equations and tables. It bypasses traditional Optical Character Recognition, achieving high accuracy while preserving semantics, making scientific knowledge stored in PDFs more accessible and usable. Nougat uses the same architecture as Donut, meaning an image Transformer encoder and an autoregressive text Transformer decoder to translate scientific PDFs to markdown, enabling easier access to them. You can use Nougat model in HuggingFace as follows:
```python
from huggingface_hub import hf_hub_download
import re
from PIL import Image

from transformers import NougatProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
import torch

processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# prepare PDF image for the model
filepath = hf_hub_download(
    repo_id="hf-internal-testing/fixtures_docvqa",
    filename="nougat_paper.png",
    repo_type="dataset",
)
image = Image.open(filepath)
pixel_values = processor(image, return_tensors="pt").pixel_values

# generate transcription (here we only generate 30 tokens)
outputs = model.generate(
    pixel_values.to(device),
    min_length=1,
    max_new_tokens=30,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
)

sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
sequence = processor.post_process_generation(sequence, fix_markdown=False)
# note: we're using repr here such for the sake of printing the \n characters, feel free to just print the sequence
print(repr(sequence))
```
Learn more about how to train and use DocVQA models in HuggingFace `transformers` library [here](https://huggingface.co/docs/transformers/tasks/document_question_answering).

## Image Captioning
![Image Captioning](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/multimodal_fusion_text_vision/image_captioning.png)
*Example of Input (Image) and Output (Text) for the Image Captioning Model. [[1]](#pretraining-paper)*
- **Inputs:**
    - Image: Image in various formats (e.g., JPEG, PNG).
    - Pre-trained image feature extractor (optional): A pre-trained neural network that can extract meaningful features from images, such as a convolutional neural network (CNN).
- **Outputs:** Textual captions: Single Sentence or Paragraph that accurately describe the content of the input images, capturing objects, actions, relationships, and overall context. See the above example for the reference.
- **Task:** To automatically generate natural language descriptions of images. This involves: (1) Understanding the visual content of the image (objects, actions, relationships). (2) Encoding this information into a meaningful representation. (3) Decoding this representation into a coherent, grammatically correct, and informative sentence or phrase.

Now, let's look at some of the popular Image Captioning models in HuggingFace:
- **ViT-GPT2:** It is a PyTorch model for generating image captions, built by combining Vision Transformer (ViT) for visual feature extraction and GPT-2 for text generation. Trained on the COCO dataset, it leverages ViT's ability to encode rich image details and GPT-2's fluency in language production to create accurate and descriptive captions. This open-source model offers an effective solution for image understanding and captioning tasks. You can use **ViT-GPT2** in HuggingFace as follows:
```python
from transformers import pipeline

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

image_to_text("https://ankur3107.github.io/assets/images/image-captioning-example.png")

# [{'generated_text': 'a soccer game with a player jumping to catch the ball '}]
```
- **BLIP-Image-Captioning:** It is a state-of-the-art image captioning model based on BLIP, a framework pre-trained on both clean and noisy web data for unified vision-language understanding and generation. It utilizes a bootstrapping process to filter out noisy captions, achieving improved performance on tasks like image captioning, image-text retrieval, and VQA. This large version, built with a ViT-L backbone, excels in generating accurate and detailed captions from images. You can use the BLIP Image Captioning model in HuggingFace as follows:
```python
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)

img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
```
- **git-base:** `microsoft/git-base` is a base-sized version of the GIT (GenerativeImage2Text) model, a Transformer decoder trained to generate text descriptions of images. It takes both image tokens and text tokens as input, predicting the next text token based on both the image and previous text. This makes it suitable for tasks like image and video captioning. Fine-tuned versions like `microsoft/git-base-coco` and `microsoft/git-base-textcaps` exist for specific datasets, while the base model offers a starting point for further customization. You can use git-base model in HuggingFace as follows:
```python
from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from PIL import Image

processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_caption)
```

Learn more about how to train and use Image Captioning models in HuggingFace `transformers` libraries [here.](https://huggingface.co/docs/transformers/v4.36.1/en/tasks/image_captioning)

## Image-Text Retrieval
![Image-Text Retrieval](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/multimodal_fusion_text_vision/image_text_retrieval.png)
*Example of Input (Text Query) and Output (Image) for the Text-to-Image Retrieval. [[1]](#pretraining-paper)*
- **Inputs:**
    - Images: Image in various formats (e.g., JPEG, PNG).
    - Text: Natural language text, usually in the form of captions, descriptions, or keywords associated with images.
- **Outputs:** 
    - Relevant images: When a text query is given, the system returns a ranked list of images most relevant to the text.
    - Relevant text: When an image query is given, the system returns a ranked list of text descriptions or captions that best describe the image.
- **Tasks:**
    - Image-to-text retrieval: Given an image as input, retrieve text descriptions or captions that accurately describe its content.
    - Text-to-image retrieval: Given a text query, retrieve images that visually match the concepts and entities mentioned in the text.

One of most popular model for the Image-Text Retrieval is CLIP.
- **CLIP (Contrastive Language-Image Pretraining):** It excels in image-text retrieval by leveraging a shared embedding space. Through contrastive learning, it pretrains on large-scale image and text datasets, enabling the model to map diverse concepts into a common space. CLIP leverages a contrastive learning approach during pretraining, where it learns to map images and text into a shared embedding space. This shared space enables direct comparison between the two modalities, allowing for efficient and accurate retrieval tasks. In image-text retrieval, CLIP can be applied by encoding images and text into the shared embedding space, and the similarity between an image and a textual query is measured by the proximity of their respective embeddings.  The model's versatility arises from its ability to grasp semantic relationships without task-specific fine-tuning, making it efficient for applications ranging from content-based image retrieval to interpreting natural language queries for images. You can use CLIP in HuggingFace for Image-Text retrieval as follows:
```python
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=image,
    return_tensors="pt",
    padding=True,
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(
    dim=1
)  # we can take the softmax to get the label probabilities
```
Learn more about how to use CLIP for Image-Text retrieval in HuggingFace [here](https://huggingface.co/docs/transformers/model_doc/clip#resources).

## Visual Grounding
![Visual Grounding](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/multimodal_fusion_text_vision/visual_grounding.jpg)
*Example of Input (Image + Text) and Output (Bounding Boxes).(a) Phrase Grounding (b) Expression Comprehension. [[1]](#pretraining-paper)*
- **Inputs:**
    - Image: A visual representation of a scene or object.
    - Natural language query: A text description or question that refers to a specific part of the image.

- **Output:** Bounding box or segmentation mask: A spatial region within the image that corresponds to the object or area described in the query. This is typically represented as coordinates or a highlighted region.
- **Task:** Locating the relevant object or region: The model must correctly identify the part of the image that matches the query. This involves understanding both the visual content of the image and the linguistic meaning of the query.

Now, see some of the popular Visual Grounding (Object Detection) models in HuggingFace.
- **OWL-ViT:** OWL-ViT (Vision Transformer for Open-World Localization) is a powerful object detection model built on a standard Vision Transformer architecture and trained on large-scale image-text pairs. It excels at "open-vocabulary" detection, meaning it can identify objects not present in its training data based on textual descriptions. By leveraging contrastive pre-training and fine-tuning, OWL-ViT achieves impressive performance in both zero-shot (text-guided) and one-shot (image-guided) detection tasks, making it a versatile tool for flexible search and identification in images. You can use OWL-ViT in HuggingFace as follows:
```python
import requests
from PIL import Image
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = [["a photo of a cat", "a photo of a dog"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process_object_detection(
    outputs=outputs, threshold=0.1, target_sizes=target_sizes
)

i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# Print detected objects and rescaled box coordinates
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}"
    )
```
- **Grounding DINO[[3]](#grounding-dino):** It combines the Transformer-based object detector (DINO) with "grounded pre-training" to create a state-of-the-art, zero-shot object detection model. This means it can identify objects even if they weren't in its training data, thanks to its ability to understand both images and human language inputs (like category names or descriptions). Its architecture combines text and image backbones, a feature enhancer, language-guided query selection, and a cross-modality decoder, achieving impressive results on benchmarks like COCO and LVIS. Essentially, Grounding DINO takes visual information, links it to textual descriptions, and then uses that understanding to pinpoint objects even within completely new categories.
You can try out the Grounding DINO model in the Google Colab [here](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/zero-shot-object-detection-with-grounding-dino.ipynb).

## Text-to-Image Generation
![Text-Image Generation](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/multimodal_fusion_text_vision/text_image_generation.png)
*Illustration of Auto-regressive and Diffusion Models for Text-Image Generation.[[1]](#pretraining-paper)*
- **Auto-regressive Models:** These models treat the task like translating text descriptions into sequences of image tokens, similar to language models generating sentences. Like puzzle pieces, these tokens, created by image tokenizers like VQ-VAE, represent basic image features. The model uses an encoder-decoder architecture: the encoder extracts information from the text prompt, and the decoder, guided by this information, predicts one image token at a time, gradually building the final image pixel by pixel. This approach allows for high control and detail, but faces challenges in handling long, complex prompts and can be slower than alternative methods like diffusion models. The generation process is shown in the above figure (a).

- **Stable Diffusion Models:** Stable Diffusion Models uses "Latent Diffusion" technique, where it builds images from noise by progressively denoising it, guided by a text prompt and a frozen CLIP text encoder. Its light architecture with a UNet backbone and CLIP encoder allows for GPU-powered image generation, while its latent focus reduces memory consumption. This unique setup empowers diverse artistic expression, translating textual inputs into photorealistic and imaginative visuals. The generation process is shown in the above figure (b).

Now, let's how can we use text-image generation models in HuggingFace.

Install `diffusers` library:
```bash
pip install diffusers --upgrade
```

In addition make sure to install transformers, safetensors, accelerate as well as the invisible watermark:
```bash
pip install invisible_watermark transformers accelerate safetensors
```
To just use the base model, you can run:
```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.to("cuda")

prompt = "An astronaut riding a unicorn"

images = pipe(prompt=prompt).images[0]
```

To learn more about text-image generation models, you can refer to the HuggingFace [Diffusers Course](https://huggingface.co/docs/diffusers/training/overview).

Now you know what some of the popular tasks and models involving Image and Text modalities. But you might be wondering on how to train or fine-tune for the above mentioned tasks. So, let's have a glimpse on training the Vision-Language models.

## Glimpse of Vision-Language Pretrained Models
![VLP Framework](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/multimodal_fusion_text_vision/vlp_framework.png)
*General framework for Transformer based vision-language models. [[1]](#pretraining-paper)*

Given an image-text pair, a VL model first extracts text features via a text encoder and a vision encoder, respectively. The text and visual features are then fed into a multimodal fusion module to produce cross-modal representations, which are then optionally fed into a decoder before generating the final outputs. An illustration of this general framework is shown in the above figure. In many cases, there are no clear boundaries among image/text backbones, multimodal fusion module, and the decoder.

Congratulations! you made it till the end. Now on to the next section for more on Vision-Language Pretrained Models.

## References
1. [Vision-Language Pre-training: Basics, Recent Advances, and Future Trends](https://arxiv.org/abs/2210.09263)<a id="pretraining-paper"></a>
2. [Document Collection Visual Question Answering](https://arxiv.org/abs/2104.14336)<a id="doc-vqa-paper"></a>
3. [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)<a id="grounding-dino"></a>
