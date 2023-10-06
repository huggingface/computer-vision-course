# LayoutLMv3: Multimodal Document Understanding

Welcome to the LayoutLMv3 documentation, a part of the LayoutLM family of models. LayoutLMv3 is a state-of-the-art multimodal model developed by the [Microsoft Document AI](https://www.microsoft.com/en-us/research/project/document-ai/overview/) team. The primary goal of this model is to tackle the complex task of understanding, extracting, and analyzing content from a wide range of documents, including PDFs, Word documents, HTML, XML, and more.

In this repository, we will focus on providing guidance and resources for using LayoutLMv3 for fine-tuning on various downstream tasks, including:

1. Sequence Classification (Image Classification)
2. Token Classification (Similar to Named Entity Recognition)
3. Visual Question Answering

For more information about other models in the LayoutLM family and their fine-tuning, please refer to the "References" section below.

## Model Background

LayoutLMv3 is built upon the Transformer architecture and incorporates both text and image embeddings to handle multimodal input. The specifics of the model architecture and pre-trained tasks may vary depending on the version. In the case of LayoutLMv3:

- The text part of the model utilizes an off-the-shelf OCR parsing tool like PyTesseract to extract text tokens and 2D word bounding box positions. For text embedding, RoBERTa word embeddings are combined with 1D and 2D position embeddings for the bounding boxes.

- The image part of the model leverages Vision Transformers (ViT) and Vision-and-Language Transformers (ViLT) to convert images into multiple image patches. These patches are then flattened and fed into the linear project features embedding.

![LayoutLMv3 Architecture](https://th.bing.com/th/id/OIP.w_g_vDT2F6TXkJQVkYX4rAHaFq?pid=ImgDet&rs=1)

The LayoutLM family of models can be fine-tuned for various downstream tasks, including:
- Sequence Classification (Document Image Classification)
- Token Classification (Similar to Named Entity Recognition)
- Visual Question Answering
- Text Extraction and Detection (OCR)

You can find more detailed information about LayoutLMv3 in the following resources:

- **LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking**
  - Research Paper: [arXiv:2204.08387](https://arxiv.org/abs/2204.08387)
  - HuggingFace Documentation: [Documentation](https://huggingface.co/docs/transformers/model_doc/layoutlmv3)
  - Model Hub: [Base](https://huggingface.co/microsoft/layoutlmv3-base) | [Large](https://huggingface.co/microsoft/layoutlmv3-large)
  - GitHub Repository: [microsoft/unilm/layoutlmv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3)

## References

For more information about other models in the LayoutLM family and their fine-tuning, please refer to the following references:

- [Other Models in the LayoutLM Family](https://github.com/NielsRogge/Transformers-Tutorials/tree/master)
- [Survey Paper on Document AI](https://arxiv.org/abs/2111.08609)

Additional LayoutLM versions:

- **LayoutLMv1: Pre-training of Text and Layout for Document Image Understanding**
  - Research Paper: [arXiv:1912.13318](https://arxiv.org/abs/1912.13318)
  - HuggingFace Documentation: [Documentation](https://huggingface.co/docs/transformers/model_doc/layoutlm)
  - Model Hub: [Base](https://huggingface.co/microsoft/layoutlm-base-uncased) | [Large](https://huggingface.co/microsoft/layoutlm-large-uncased)
  - GitHub Repository: [microsoft/unilm/layoutlm](https://github.com/microsoft/unilm/tree/master/layoutlm)

- **LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding**
  - Research Paper: [arXiv:2012.14740](https://arxiv.org/abs/2012.14740)
  - HuggingFace Documentation: [Documentation](https://huggingface.co/docs/transformers/model_doc/layoutlmv2)
  - Model Hub: [Base](https://huggingface.co/microsoft/layoutlmv2-base-uncased) | [Large](https://huggingface.co/microsoft/layoutlmv2-large-uncased)
  - GitHub Repository: [microsoft/unilm/layoutlmv2](https://github.com/microsoft/unilm/tree/master/layoutlmv2)

- __LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding__
    - Research Paper: [arXiv:2104.08836](https://arxiv.org/abs/2104.08836)
    - HuggingFace Documentation: [Documentation](https://huggingface.co/docs/transformers/model_doc/layoutxlm)
    - Model Hub: [Base](https://huggingface.co/microsoft/layoutxlm-base)
    - GitHub: [microsoft/unilm/layoutxlm](https://github.com/microsoft/unilm/tree/master/layoutxlm)

- __LayoutReader: Pre-training of Text and Layout for Reading Order Detection__
    - Research Paper: [arXiv:2108.11591v2](https://arxiv.org/abs/2108.11591v2)
    - HuggingFace Documentation: N/A
    - Model Hub: [Nielsr Model Card](https://huggingface.co/nielsr/layoutreader-readingbank)
    - GitHub: [microsoft/unilm/layoutreader](https://github.com/microsoft/unilm/tree/master/layoutreader)