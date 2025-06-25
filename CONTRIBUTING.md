# How To Contribute

Hey 👋, great that you want to contribute to the Community Computer Vision Course! We're happy to hear your suggestions and ideas!

### Adding content to the course
*Important Note: If you’ve never contributed to open-source projects on GitHub, kindly read [this document](https://www.notion.so/19411c29298644df8e9656af45a7686d?pvs=25) which shows to do so with an example for the skops repository.*

1. First go to the [discussion section](https://github.com/huggingface/computer-vision-course/discussions/).
2. Here you find a section for each unit of the course. Go to the unit you want to contribute to. Open a new discussion and describe what you want to add.
3. Wait for an approval from the repository maintainers or change requests
4. When your suggestions is approved, follow these steps:
   1. Create an `.mdx` file or Jupyter Notebook for the topic you want to contribute to
   2. Please carefully read through our [Content Guidelines](#📝-content-guidelines)
   3. When you feel like you are ready, create a pull request to this repository. See [How to Create a Pull Request](#how-to-create-a-pull-reques)


### Typos/Bug fixes
1. Open an [Issue](https://github.com/huggingface/computer-vision-course/issues) describing which content you want to add, change or fix
2. Wait for an approval from the repository maintainers
3. Follow the steps below to create a PR


### How to Create a Pull Request (PR)
1. Fork the repository
2. Create a new branch for your changes
3. Make your changes
4. Create a pull request to the [stage](https://github.com/huggingface/computer-vision-course/tree/stage) branch of the main repository
5. Wait for the maintainers to approve your PR
6. Celebrate your contribution 🥳🎉

### Releases
We will collect contributions in the *stage* branch and publish new releases of the course in regular time intervals.


## 📝 Content Guidelines

**Syntax and Doc Rules ❗️❗️**

These rules are required to render the course on hf.co/learn 😊 
1. Every chapter should have a main header (h1, e.g. # Introduction) before the content. 
2. We can use the syntax features in Hugging Face course! This includes 👇 

**Tip Blocks**
Write tips like so:
```
<Tip>
Write your note here
</Tip>
```
You can write warnings like this: 
`<Tip warning={true}>`

**Framework Dependent Code**
To have multiple frameworks in one code snippet with a toggle, you can do:
```
<frameworkcontent>
<pt>
PyTorch content goes here
</pt>
<tf>
TensorFlow content goes here
</tf>
<flax>
Flax content goes here
</flax>
</frameworkcontent>
```
**Embedding Gradio Demos**
You can embed any Gradio demo that is hosted on Hugging Face Spaces like below 👇 Just provide the `src` the url `{{username}}-{{space_id}}.hf.space`.
```
<iframe
	src="https://openai-whisper.hf.space"
	frameborder="0"
	width="850"
	height="450">
</iframe>
```

**Anchor Links for Headers**
If you want to refer to a section inside the text, you can do it like below 👇 
```
## My awesome section[[some-section]]
// the anchor link is: `some-section`
```

**Code Blocks**
You can write codeblocks by wrapping it with three backticks. Please add the associated language code, e.g. `py` or `bash` after top backticks to enable language specific rendering of code blocks. 

**LaTeX**
You can write in-line LaTeX by writing it like this: ` \\( X )\\`
You can write stand alone LaTeX by enclosing with  `$$`.
For example 👇 
```
$$Y = X * \textbf{dequantize}(W); \text{quantize}(W)$$
```


3. Add your chapter to `_toctree.yml`.

- Note that the directory structure is as follows, so when you add a new chapter, make sure to stick to it:
```
.
└── course/
    └── chapters/
        ├── chapter0/
        │   ├── introduction.mdx
        │   └── getting_started.mdx
        ├── chapter1/
        │   └── ...mdx
        └── _toctree.yml
```

- If you need advice on the tone of your content, feel free to check out [Hugging Face Audio Course](https://huggingface.co/learn/audio-course/chapter0/introduction) as it's a good example.
  
- If you have any images, videos and more in your PRs, please store them in [this Hugging Face repository](https://huggingface.co/datasets/hf-vision/course-assets) to keep this repository lightweight. You can ask for an access to the organization if you aren't a part of it yet. The steps to do so are below 👇
1. Request to join the https://huggingface.co/hf-vision organization. 
2. Upload an image to https://huggingface.co/datasets/hf-vision/course-assets, e.g. via the web UI.
3. Get the URL (e.g. if there's blob in the link replace with resolve https://huggingface.co/datasets/huggingface-course/audio-course-images/resolve/main/all_models.png) right click to "Download" button and copy the link.
4. Use that in standard markdown like ![image](link-to-image)

### Notebooks

Thanks to Hugging Face's documentation builder, when we add `[[open-in-colab]]` on top of `.mdx` files, it creates a button that you can automatically open a notebook containing the notebook version of your markdown file. If you still want to create a notebook that is separate from markdown (and the markdown is complete by means of context you're providing in the notebook), you can do so, in the notebooks folder. Make sure to add it to the associated chapter's folder under notebooks (and if it doesn't exist, feel free to create it).


### 🗣 Asking for Help

Do not hesitate to ask for help in #cv-community-project channel on the Hugging Face discord. 🫂 

### Tips and Hints


- For an easier collaboration when **working on notebooks together**, feel free to use [ReviewNB](https://www.reviewnb.com/), which is free for open-source and educational use cases.

- In the requirements.txt file you can find some packages that can be helpful when creating the material. As we're originating from the HuggingFace community, we can recommend using the  [transformers](https://huggingface.co/docs/transformers/), [datasets](https://huggingface.co/docs/datasets/), [evaluate](https://huggingface.co/docs/evaluate/) and [timm](https://huggingface.co/docs/timm/) libraries.
