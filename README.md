![Course](https://github.com/johko/computer-vision-course/assets/53175384/58e39903-5a3a-4d48-8f3c-5811f31b93b5)

# Community-led Computer Vision Community Course Sprint 🤗

This is the repository for a community-led course on Computer Vision. Once finished, the course will be released in huggingface.co/learn. Below, you can find how you can help us in this effort.

### Table of Contents

- 🏃‍♂️ **How to get started**
- 🤝 **How to contribute to course**
- 📆 **Deadlines**
- 💝 **Prizes**
- 📝 **Content guidelines (Syntax, Notebooks, Becoming a Reviewer)**
- 🫂 **Asking for Help**

### 🏃‍♂️ How to get started

1. Join us in Discord 👾

   Join [the Hugging Face discord](https://discord.gg/hugging-face-879548962464493619), take the role open-source and join us at the channel #cv-community-project.
<img width="491" alt="image" src="https://github.com/lunarflu/fork-computer-vision-course/assets/70143200/c13d5b34-ed1c-4f12-b044-192484b94f9d">
<img width="180" alt="image" src="https://github.com/lunarflu/fork-computer-vision-course/assets/70143200/b3372a47-711f-4b43-bc85-0ba2b6f8b914">

2. Pick a section from the [table of contents](https://docs.google.com/spreadsheets/d/1fjmbsdGwe7IUMBv74LDC7IpoJy8ijiFdzGdnDlBv6eA/edit#gid=0) and add your first name and discord username there.
(Please only sign up for two chapters maximum for now.)
3. Connect with your team members in Discord

   


### 🤝 How to contribute to the course
Important Note: If you’ve never contributed to open-source projects on GitHub, kindly read [this document](https://www.notion.so/19411c29298644df8e9656af45a7686d?pvs=25) to learn how to do so.

Before you, start reading more on the contribution guidelines, this is how our course outline looks like:

```
.

└── chapters/
	├── chapter0/
        │   ├── introduction.mdx
        │   └── getting_started.mdx
        ├── chapter1/
        │   └── ...mdx
        └── _toctree.yml
```

1. First go to the [discussion section](https://github.com/johko/computer-vision-course/discussions/).
2. Each new chapter outline should be put up under a pinned discussion. As shown in this [example](https://github.com/johko/computer-vision-course/discussions/80). 
3. Each chapter can contain multiple topics and sub-topics. All of that should be defined in the chapter outline. Also, for each sub-topic, a single contributor should be assigned to work on that. See more in this [example](https://github.com/johko/computer-vision-course/discussions/80).
4. Interested contributors motivated to work in a common chapter can come together to form a team. Where each contributor can directly contribute to a sub-topic and other fellow teammates can act as reviewers.
5. One contributor from each team should fork the repository and other contributors should mutually agree on working collaboratively under that forked repo. 
6. Under the forked repo, contributors should create issues (that will be the sub-topic name) and start working on that issue. 
7. When contributors work on a subtopic, they should follow the below instructions:
   1. Create `.mdx` files or Jupyter Notebooks for the sub-topics you want to contribute to
   2. Make sure to update the requirements.txt file in the root of the repository
   3.  When you feel like you are ready, create a pull request to this repository. 
   4.  Your teammates will review your PR under that forked repo and then if things get approved, you can create the PR to merge it in our main branch 🤗

   
**Tip:** Contribute one subsection at a time, so that it’s not overwhelming for both you and reviewers.


## 📆 Deadlines

We are aiming to have the version 1 of the course with this sprint. 

**Sprint Beginning:** Dec 3rd

**Submission for Pull Request Reviews Deadline:** Dec 13th

**Iterating over Pull Reviews until:** Dec 27th

First version will be complete before December 29th.

## 💝 Prizes

We will have prizes for those who contribute to the course 🤗 
Moreover, each contributor will have their name added to credits section of the associated chapter.

- We will provide one month of Hugging Face PRO subscription or GPU grant.
- You can earn the special merch made for this sprint (with the logo of this sprint). 👕👚

The amount of contribution required to earn the prizes will be announced. Stay tuned!

  
### 📝 Content Guidelines

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
You can write LaTeX by writing it like this: `$$...$$``
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
  
- Before contributing, please read the general [contribution guide](https://huggingface2.notion.site/Contribution-Guide-19411c29298644df8e9656af45a7686d?pvs=4).

- If you have any images, videos and more in your PRs, please store them in [this Hugging Face repository](https://huggingface.co/datasets/hf-vision/course-assets) to keep this repository lightweight. You can ask for an access to the organization if you aren't a part of it yet. The steps to do so are below 👇
1. Request to join the https://huggingface.co/hf-vision organization. 
2. Upload an image to https://huggingface.co/datasets/hf-vision/course-assets, e.g. via the web UI.
3. Get the URL (e.g. https://huggingface.co/datasets/huggingface-course/audio-course-images/blob/main/all_models.png) right click to "Download" button and copy the link.
4. Use that in standard markdown like ![image](link-to-image)

### Notebooks

Thanks to Hugging Face's documentation builder, when we add `[[open-in-colab]]` on top of `.mdx` files, it creates a button that you can automatically open a notebook containing the notebook version of your markdown file. If you still want to create a notebook that is separate from markdown (and the markdown is complete by means of context you're providing in the notebook), you can do so, in the notebooks folder. Make sure to add it to the associated chapter's folder under notebooks (and if it doesn't exist, feel free to create it).

### Become a Reviewer

Is everything already assigned, but do you want to contribute to the course? No worries, you can still become a reviewer! This will allow you to review the notebooks and READMEs and give feedback to the authors.

1. Go [here](https://docs.google.com/spreadsheets/d/1fjmbsdGwe7IUMBv74LDC7IpoJy8ijiFdzGdnDlBv6eA/edit#gid=0) and at the bottom choose the "Reviewers" tab,
2. Add your first name and GitHub username,
3. You'll be added to the repo as a contributor and will be able to review Pull Requests (PRs)!
4. Assign yourself to any PRs you think make sense for you,
5. While reviewing, ask yourself if the changes and additions make sense - After all, the most important part of a course is conveying ideas properly, which makes the learning experience smoother.
6. Share your feedback and ideas with the authors on how they can improve. In the long term, we're aiming to make a robust, high-quality course accessible to everyone!
7. Finally, every PR can be merged when it has two approvals from reviewers.

### 🗣 Asking for Help

Do not hesitate to ask for help in #cv-community-project channel. 🫂 

### Tips and Hints


- For an easier collaboration when **working on notebooks together**, feel free to use [ReviewNB](https://www.reviewnb.com/), which is free for open-source and educational use cases.

- In the requirements.txt file you can find some packages that can be helpful when creating the material. As we're originating from the HuggingFace community, we can recommend using the  [transformers](https://huggingface.co/docs/transformers/), [datasets](https://huggingface.co/docs/datasets/), [evaluate](https://huggingface.co/docs/evaluate/) and [timm](https://huggingface.co/docs/timm/) libraries.
