# Transfer Learning of Multimodal Models

In the preceding sections, we've delved into the fundamental concepts of multimodal models such as CLIP and its related counterparts. In this chapter, we will try to find out how you can use different types of multimodal models for your tasks.

There are several approaches to how you can adapt multimodal models to your use case:

1. **Zero\few-shot learning**. Zero\few-shot learning involves leveraging large pretrained models capable of solving problems not present in the training data. These approaches can be useful when there is little labeled data for a task (5-10 examples) or there is none at all. [Unit 11](https://huggingface.co/learn/computer-vision-course/unit11/1) will delve deeper into this topic.

2. **Training the model from scratch**. When pre-trained model weights are unavailable or the model's dataset substantially differs from your own, this method becomes necessary. Here, we initialize model weights randomly (or via more sophisticated methods like [He initialization](https://arxiv.org/abs/1502.01852)) and proceed with the usual training. However, this approach demands substantial amounts of training data.

3. **Transfer learning**. Transfer learning, unlike training from scratch, uses the weights of the pretrained model as initial weights.

This chapter primarily focuses on the transfer learning aspect within multimodal models. It will recap what transfer learning entails, elucidate its advantages, and provide practical examples illustrating how you can apply transfer learning to your tasks!

## Transfer learning

More formally, transfer learning is the set of machine learning techniques in which knowledge, representations or patterns that are obtained from solving one problem are reused to solve another, but similar problem.

In the context of deep learning, in transfer learning, when training a model for a particular task, we use the weights of another model as the initial weights. The pretrained model has typically been trained on a huge amount of data and has useful knowledge about the nature and relationships in that data. This knowledge is embedded in the weights of this model, and therefore if we use them as initial weights, we transfer the knowledge embedded in the pretrained model into the model we are training.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/multimodal_trasnsfer_learning_images/transfer_learning_light.png" alt="Transfer Learning">
</div>

This approach has several advantages:

- **Resource Efficiency:** Since the pretrained model was trained on a huge amount of data over a long period, transfer learning requires much less computing resources for model convergence.

- **Reducing the size of labeled data:** For the same reason, less data is required to achieve decent quality on the test sample.

- **Knowledge Transfer:** When fine-tuning to the new task, the model capitalizes on the pre-existing knowledge encoded within the pre-trained model's weights. This integration of prior knowledge often leads to enhanced performance on the new task.

However, despite its advantages, transfer learning has some challenges that should also be taken into account:

- **Domain Shift:** Adapting knowledge from the source domain to the target domain can be challenging if the data distributions differ substantially.

- **Catastrophic forgetting:** During fine-tuning process, the model adjusts its parameters to adapt to the new task, often causing it to lose the previously learned knowledge or representations related to the initial task.

## Transfer Learning Applications

We'll explore practical applications of transfer learning across various tasks. The table below provides a description of the tasks that can be solved using multimodal models, as well as examples of how you can fine-tune them on your data.

| Task        | Description                                                      | Model                                             |
| ----------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| [Fine-tune CLIP](https://colab.research.google.com/github/fariddinar/computer-vision-course/blob/main/notebooks/Unit%204%20-%20Multimodal%20Models/Clip_finetune.ipynb)| Fine-tuning CLIP on a custom dataset | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) |
| [VQA](https://huggingface.co/docs/transformers/main/en/tasks/visual_question_answering#train-the-model) | Answering a question in natural <br/> language based on an image | [dandelin/vilt-b32-mlm](https://huggingface.co/dandelin/vilt-b32-mlm) |
| [Image-to-Text](https://huggingface.co/docs/transformers/main/en/tasks/image_captioning) | Describing an image in natural language | [microsoft/git-base](https://huggingface.co/microsoft/git-base) |
| [Open-set object detection](https://docs.ultralytics.com/models/yolo-world/) | Detect objects by natural language input |  [YOLO-World](https://huggingface.co/papers/2401.17270) |
| [Assistant (GTP-4V like)](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#train) | Instruction tuning in the multimodal field | [LLaVA](https://huggingface.co/docs/transformers/model_doc/llava) |
