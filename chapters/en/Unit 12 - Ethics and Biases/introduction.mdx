# Introduction

Welcome to the introduction chapter for the unit Ethics and Bias in Computer Vision. This chapter will build the foundation for many important concepts we will encounter later in this unit.
In this chapter, we will:
- Cover the popular ImageNet Roulette case study in context to ethics and bias in computer vision with examples.
- Explore what implications it can have on people and certain groups.
- Take a look into the consequences of the experiment.
- Efforts by ImageNet team to tackle and mitigate these problems.
- Conclude the chapter with some questions on the case study and lay foundation for next chapters.

So let us dive in 🤗

## ImageNet Roulette: A Case Study on Biases in Classification 

Imagine you wake up Sunday morning and play around with your phone. You come across this app that tries to return sarcastic and funny labels if you upload different images or maybe take a selfie. You don't mind some fun, so you try the app out by uploading a selfie, and to your shock, it returns an alarming label. It mentions you as a crime suspect (this crime can also be highly dangerous and heinous). You also see social media posts on the same app of different people with provoking labels, increasing the chances of racial and gender profiling. Some of these labels would mean a person who is a crime offender, a person with specific facial features linked with one's ethnicity, or a person's ancestry. This app returns very offensive labels often and might harm self-interests and target a special group of people. A wide range of such labels, which can offend people based on their religion, ethnicity, gender, or age, were present in the app, and you are just shocked and also kind of confused about what's going on with this. 

AI has made our lives easier and more comfortable, but many times, if AI is not kept in check, it can cause havoc in people's lives. Humans should be more inclusive and aware of others' needs and preferences. The same human values must be incorporated and reflected while developing and deploying AI models. AI models should not create negative sentiments or try to manipulate anyone against a group. 

### Introduction to ImageNet: A Large Scale Dataset for Object Recognition

ImageNet is a large-scale dataset that was created for object recognition benchmarks at scale. The aim was to map out the entire world of objects to make machines around us smarter in scene understanding, in which humans are far better. This dataset was one of the earliest attempts of its kind to create a large-scale dataset for object recognition. 

The ImageNet team started scraping image data from various sources on the internet. The original dataset consisted of around 14,197,122 images with 21,841 classes; this was referred to as Imagenet-21K, reflecting around 21K classes. The annotations were crowdsourced using Amazon Mechanical Turk. A smaller subset of this dataset called ImageNet-1K contained 1,281,167 training images, 50,000 validation images, and 100,000 test images with 1000 classes which was used as the foundation for the popular ImageNet Large Scale Visual Recognition Challenge (ILSVRC). The ILSVRC became the competition ground for many aspiring companies and labs working in computer vision to outperform previous approaches on accurately labeling objects. The structure for ImageNet was based on WordNet, a database of word classifications developed at Princeton University.

<Tip>

💡You can read more about ImageNet [here](https://www.image-net.org/). Also check out [this TED Talk](https://www.youtube.com/watch?v=40riCqvRoMs) by Prof. Fei Fei Li on the same topic.
</Tip>

### Motivation Behind ImageNet Roulette 🃏

Now, let's talk about the app we discussed earlier. This app existed on the internet a few years ago as a web application. This was an intentional experiment termed ImageNet Roulette and is still one of the most popular cases of how AI models can go wrong if the training data is not prepared carefully with guidelines. This project was developed by artist Trevor Paglen and researcher Kate Crawford. They trained their model based on the 2,833 subcategories of the Person category found in the dataset.

The model was trained using Caffe on the images and labels in the "person" category. The app prompts the user for an image, and a face detector detects faces in the image. The detected face was then sent to the Caffe model for classification. Finally, the app returned an image with a bounding box around the detected face and the label the Caffe model predicted. 

The main motivation behind ImageNet Roulette was to show inherent biases in classifying people. It was trained on only the "person" category from the ImageNet dataset (as discussed earlier). So what went wrong? The inference on different images reflected harmful and provocative on many levels. The biggest cause for bias in this system was largely due to the already available ImageNet categories. The classes reflect how they were kind of absurd, offensive, and provocative while annotating the images. Some of the labels (after rephrasing them from their initial labels to avoid triggering) would mean a person who is an addict, a person with questionable character, a person who is against a specific group of people, a person who is unsuccessful, and a loser, and many more. 

A wide range of labels that categorize people based on gender, race, profession, etc., was inherent in ImageNet. And where did it go wrong? These labels are all drawn from the structure of WordNet. This is exactly where the biases crept into this model (due to overlooking the data preparation process, irrelevant images were downloaded in bulk). We will explore the reasons mentioned by the ImageNet team `later` in the chapter.

Would you like such models to be deployed without any checks? If deployed, are you fine with letting people around you refer to you as an unsuccessful person and make a viral post? This is what went wrong and unlooked for while preparing the dataset. 

### Implications of ImageNet Roulette

Let us explore the implications this experiment had:

1. It exposed deep rooted biases within the ImageNet annotations which are often offensive and stereotypical, particularly concerning race and gender.
2. The experiment also questions the integrity of datasets used to train AI models especially within the ImageNet dataset. It highlights the need for more rigorous scrutiny and ethical considerations in creating and annotating the training data.
3. The shocking results acted as a catalyst for discussions around ethical considerations in AI. It prompted a broader conversation within the AI community about the responsibility to ensure fair and unbiased training data, emphasizing the need for ethical data practices.
In general, if models like this are deployed in real life applications, it can have alarming effects on different people and target groups.

### Consequences of ImageNet Roulette

Initially, the "person" category was not noticed as ImageNet was an object recognition benchmark. But after this experiment, some crucial changes happened in the community. In this case study, the creators were able to show the problems with the inherent biases in ImageNet (which was in shadow till around 2018, when some research started showing up). After a few days, ImageNet released a research paper summarizing their one-year-long project funded by NSF. The full ImageNet dataset was disabled for download since January 2019, whereas the 1000-category dataset ImageNet-1K was not affected. The ImageNet team provided some underlying issues and ways to deal with them (surprisingly, the ImageNet Roulette was not mentioned in their report).

**Problem 1: Offensive Synonym Sets in WordNet** WordNet contains many offensive synsets that are inappropriate as image labels. Somehow, a lot of these labels creeped into ImageNet and were included. 

**Solutions:** 
a. ImageNet appointed a group of in-house manual annotators to classify synsets in three categories: *offensive*, *sensitive* and *safe*. Offensive labels are racial or gender slurs, sensitive labels are not offensive but might cause offense according to the context, and safe labels are not offensive.
b. Out of the 2,832 synsets within the person category, 1,593 unsafe synsets (offensive and sensitive) were identified and the remaining 1,239 synsets were temporarily deemed safe. 
c. A new version of ImageNet was prepared by removing unsafe synsets resulting in removal of around 600,000 images in total.

**Problem 2: Non-imageable concepts** Some synsets might not be offensive, but including them in the overall dataset is also not logical. For example, we cannot classify a person in the image as a philanthropist. Similarly, there might be many synsets that cannot be captured visually using images. 

**Solutions:**

a. For such concepts, multiple workers were asked to rate each of the 2,394 people synsets (safe + sensitive).
b. The rating was done based on the ease with which the synset arouses mental imagery in a scale of 1-5, 1 being very hard and 5 begin very easy.
c. The median rating was 2.36 and around 219 synsets had a rating more than 4, images with very low imageability were removed.

**Problem 3: Diversity of images** Many images in ImageNet might have lesser representation. For example, an image search of a particular profession can end up returning different gender ratios as in the real world. Images of construction workers or gangsters might be more inclined toward a particular gender or race.  Not only during search but also during annotations and data cleaning, annotators might be inclined to respond to particular categories in an already socially stereotypical manner. 
**Solutions:**
a. To mitigate such stereotypes in search and annotations, images should have higher visual arousal (visually more strong).
b. ImageNet team did a demographic analysis on most imageable attributes like gender, color and age.
c. After this analysis, the dataset was balanced by removing overrepresented attributes in a synset leading to a more uniform gender, color and age balance. 

**Problem 4: Privacy Concerns** While classification was subject to some inherent biases, to protect an individual's identity, privacy is also an equally important factor. If these classifications from the experiment were viral, it would have a huge impact on people's lives and overall well-being. To ensure this, AI models should not only be fair but also preserve subjects' privacy.

**Solutions:**

a. Imagenet-1K dataset had 3 people categories. Separate face annotations were carried out and a face-blurred version of the dataset was created.
b. Image obfuscation techniques like blurring and mosaicing were applied to these images.
c. It was shown that these images lead to a very minimal drop of accuracy while benchmarking on object recognition tasks and are suitable for training privacy-aware visual classifiers.

<Tip>

💡For more details on the ImageNet Roulette experiment, you can follow the article on ImageNet Roulette. The experiment is posted on [Excavating AI](https://excavating.ai/) which discusses this in detail. To know more about ImageNet's stance and research on mitigating these issues, you can take a look into the full technical report submitted by them [here](https://www.image-net.org/filtering-and-balancing/).
</Tip>

## Conclusion 

In the later chapters, we will also follow the same flow for case studies and try to answer some basic questions. Although we will discuss AI models in general, our focus will be mainly on CV models and the ethical concerns revolving around them.

1. Exploration, what the case study or experiment is all about?
2. What can go wrong or what went wrong and where?
3. What is the impact on target groups and other implications (impact assessment)?
4. How to evaluate bias in CV models using metrics?
4. How to mitigate these problems for fair and ethical development of CV models.
5. Role of community and other target groups in fostering and cultivating open dialogues.

In conclusion over the whole unit, we will come across various case studies related to ethics and bias, will evaluate bias and try to think of what impact they can have if biases are unresolved. We will also explore various strategies to mitigate the biases, and make CV models safe and inclusive for usage.
