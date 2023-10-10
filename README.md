# Computer Vision Community Course

This is the repository for a community-led course on Computer Vision. Below, you can find how you can help us in this effort. 


### How to get started

1. Join us in Discord 👾

   Join [the Hugging Face discord](hf.co/join/discord), take the role open-source and join us at the channel #cv-community-project. 
<img width="491" alt="image" src="https://github.com/lunarflu/fork-computer-vision-course/assets/70143200/c13d5b34-ed1c-4f12-b044-192484b94f9d">
<img width="180" alt="image" src="https://github.com/lunarflu/fork-computer-vision-course/assets/70143200/b3372a47-711f-4b43-bc85-0ba2b6f8b914">



2. Pick a section from the [table of contents](https://docs.google.com/spreadsheets/d/1fjmbsdGwe7IUMBv74LDC7IpoJy8ijiFdzGdnDlBv6eA/edit#gid=0) and add your first name and discord username there. 
(Please only sign up for two chapters maximum for now.)
3. Connect with your team members in Discord


### How to contribute to the course

1. One person from your team should create a fork of this repository
2. Add your team members as collaborators to the fork
3. Add a branch for your section
4. Create Jupyter Notebooks for the chapters you want to contribute to
5. Add the notebooks to the fork
6. Work on the notebooks in your team and collaborate via standard git functionalities
7. Make sure to update the requirements.txt file in the root of the repository
8. When you feel like you are ready, create a pull request to this repository

### Syntax and Doc Rules ❗️❗️

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

### Become a Reviewer

Is everything already assigned, but you still really want to contribute to the course? No worries, you can still become a reviewer! This will allow you to review the notebooks and READMEs and give feedback to the authors.

1. Go [here](https://docs.google.com/spreadsheets/d/1fjmbsdGwe7IUMBv74LDC7IpoJy8ijiFdzGdnDlBv6eA/edit#gid=0) and at the bottom choose the "Reviewers" tab,
2. Add your first name and GitHub username,
3. You'll be added to the repo as a contributor and will be able to review Pull Requests (PRs)!
4. Assign yourself to any PRs you think make sense for you,
5. While reviewing, ask yourself if the changes and additions make sense - After all, the most important part of a course is conveying ideas properly, which makes the learning experience smoother.
6. Share your feedback and ideas with the authors on how they can improve. In the long term, we're aiming to make a robust, high-quality course accessible to everyone!
7. Finally, every PR can be merged when it has two approvals from reviewers.
   

### Tips and Hints

- Before contributing, **if it's your first time**, please read the general [contribution guide](https://huggingface2.notion.site/Contribution-Guide-19411c29298644df8e9656af45a7686d?pvs=4).

- If you have any **bigger assets like datasets, models, custom demos**, please store them in [this Hugging Face repository](https://huggingface.co/datasets/hf-vision/course-assets) to keep this repository lightweight. You can ask for an access to the organization if you aren't a part of it yet.

- For an easier collaboration when **working on notebooks together**, feel free to use [ReviewNB](https://www.reviewnb.com/), which is free for open-source and educational use cases.

- In the requirements.txt file you can find some packages that can be helpful when creating the material. As we're originating from the HuggingFace community, we can recommend using the  [transformers](https://huggingface.co/docs/transformers/), [datasets](https://huggingface.co/docs/datasets/), [evaluate](https://huggingface.co/docs/evaluate/) and [timm](https://huggingface.co/docs/timm/) libraries.
