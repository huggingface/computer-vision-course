# Transfer Learning / Fine Tuning for Multimodal Models

In this directory, you can find multiple notebooks that introduce how to fine-tune Multimodal models to downstream task like:
 - Image-to-Text / Text-to-Image
 - Document Classification
 - Visual Document Question Answering 
 - Form Understanding, etc.

We use different types of Multimodal Models ranging from:
 - LayoutLM family
 - DETR
 - DiT, etc.

This includes also the setup for fine-tuning and inferencing at the end. Feel free to use them as sources for your project.

For Multimodal model related to Text and Image, feel free to start with this [paper](https://arxiv.org/abs/2111.08609).

# Model Introduction

## LayoutLM Family
LayoutLM is a multimodal model that is develop by the [Microsoft Document AI](https://www.microsoft.com/en-us/research/project/document-ai/overview/) team where they most focusing on develop techniques and models to solve the problem of understanding, extracting and analysing complex document such as PDF, Word Document, HTML, XML, etc.

The backbone of the model is Transformers architecture with Text Embedding to handle text input and Image Embedding to handle image input. Depend on the version, the embedding techniques and pre-trained task might be slightly different.

The model can fine-tune downstream various tasks:
- Sequence Classification (Document Image Classification)
- Token Classification (Similar like Named Entity Recognition)
- Visual Question Answering (Talk with your Document)

![LayoutLMv2 Architecture](https://th.bing.com/th/id/OIP.uS8KaZ-dipmmY57gkEReZgHaGm?pid=ImgDet&w=897&h=800&rs=1)

For more paper related to LayoutLM family:
- __LayoutLMv1: Pre-training of Text and Layout for Document Image Understanding__
    - Research Paper: [arXiv:1912.13318](https://arxiv.org/abs/1912.13318)
    - HuggingFace Documentation: [Documentation](https://huggingface.co/docs/transformers/model_doc/layoutlm)
    - Model Hub: [Base](https://huggingface.co/microsoft/layoutlm-base-uncased) | [Large](https://huggingface.co/microsoft/layoutlm-large-uncased)
    - GitHub: [Repository](https://github.com/microsoft/unilm/tree/master/layoutlm)

- __LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding__
    - Research Paper: [arXiv:2012.14740](https://arxiv.org/abs/2012.14740)
    - HuggingFace Documentation: [Documentation](https://huggingface.co/docs/transformers/model_doc/layoutlmv2)
    - Model Hub: [Base](https://huggingface.co/microsoft/layoutlmv2-base-uncased) | [Large](https://huggingface.co/microsoft/layoutlmv2-large-uncased)
    - GitHub: [Repository](https://github.com/microsoft/unilm/tree/master/layoutlmv2)

- __LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking__
    - Research Paper: [arXiv:2204.08387](https://arxiv.org/abs/2204.08387)
    - HuggingFace Documentation: [Documentation](https://huggingface.co/docs/transformers/model_doc/layoutlmv3)
    - Model Hub: [Base](https://huggingface.co/microsoft/layoutlmv3-base) | [Large](https://huggingface.co/microsoft/layoutlmv3-large)
    - GitHub: [Repository](https://github.com/microsoft/unilm/tree/master/layoutlmv3)

- __LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding__
    - Research Paper: [arXiv:2104.08836](https://arxiv.org/abs/2104.08836)
    - HuggingFace Documentation: [Documentation](https://huggingface.co/docs/transformers/model_doc/layoutxlm)
    - Model Hub: [Base](https://huggingface.co/microsoft/layoutxlm-base)
    - GitHub: [Repository](https://github.com/microsoft/unilm/tree/master/layoutxlm)

- __LayoutReader: Pre-training of Text and Layout for Reading Order Detection__
    - Research Paper: [arXiv:2108.11591v2](https://arxiv.org/abs/2108.11591v2)
    - HuggingFace Documentation: N/A
    - Model Hub: [Nielsr Model Card](https://huggingface.co/nielsr/layoutreader-readingbank)
    - GitHub: [Repository](https://github.com/microsoft/unilm/tree/master/layoutreader)