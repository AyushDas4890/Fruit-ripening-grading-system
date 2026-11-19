# Fruit Ripeness Grading via Ordinal Regression
### Presenter Script — 13 Slides

### Slide 1 — Introduction

Good morning everyone.

Today, I am presenting my project, Fruit Ripeness Grading via Ordinal Regression.

The objective is to classify strawberries and avocados into four stages: unripe, partially ripe, ripe and rotten.

I have used deep learning and computer vision to build this system. The interesting part is that I use ordinal regression because these four stages have a natural order.

---

### Slide 2 — Agenda

I'll divide my presentation into seven parts.

First, I'll explain the problem and why the order of ripeness matters. Then I'll discuss the dataset, preprocessing and how I prevented data leakage.

After that, I'll explain ordinal regression, the CORAL architecture, model training, experimental results, error analysis, and finally the findings and limitations.

---

### Slide 3 — Ripeness Is a Ladder, Not a List

Why do we need fruit ripeness grading?

In sorting facilities, fruits need to be sorted based on whether they should be sold, stored or discarded.

The important thing is that ripeness stages are ordered.

For example, if the actual grade is rotten and the model predicts ripe, it is one grade wrong. But predicting unripe is three grades wrong.

These mistakes are not equally serious. Normal classification treats them as incorrect predictions, which motivates our use of ordinal regression.

---

### Slide 4 — Real Photos, Two Fruits, Four Stages

For this project, I used a publicly available dataset from Mendeley Data, published by Kamat and colleagues in 2024.

The images contain real strawberries and avocados collected from farms and markets in Maharashtra.

As shown in the grid, both fruits have four ripeness stages, from unripe to rotten.

The dataset contains 758 labelled photographs and 924 individual fruit annotations because some photographs contain multiple fruits.

So, we have eight classes in total: two fruit types multiplied by four ripeness stages.

---

### Slide 5 — From Detection Boxes to Crops

The original dataset provides annotations in YOLO format.

Each annotation contains a class ID and four bounding-box values: centre X, centre Y, width and height.

I use these coordinates to locate individual fruits and crop them from the original images.

I also add an eight-percent border around each bounding box.

Finally, the crops are resized to 224 by 224 pixels and normalized using ImageNet statistics so they can be passed into our models.

---

### Slide 6 — Same Fruit Must Never Sit in Train and Test

This slide explains how I prevented data leakage.

Some photographs are burst shots or edited copies of the same fruit. If one copy is used for training and another for testing, the model may give artificially high results.

I first tried two approaches, but they had problems. One merged 93 different avocados into a single group, while a normal 70/15/15 split left only two ripe avocados in the test set.

The final approach groups images using their source photographs, nearby photo numbers and visual similarity.

This creates 159 groups, which are divided using five-fold grouped cross-validation. Every crop is tested exactly once.

---

### Slide 7 — Ask Three Yes/No Questions

Now we come to the main idea of ordinal regression.

Instead of asking the model to directly select one of four grades, we ask it three questions.

Is the grade greater than zero? Is it greater than one? And is it greater than two?

As shown in the table, grade two gives the answers yes, yes and no, represented as [1,1,0].

The predicted grade is simply the number of yes answers.

This makes larger grading mistakes affect more answers than smaller mistakes.

---

### Slide 8 — The CORAL Head

This slide shows how our ordinal model works internally.

First, the 224 by 224 image goes into ResNet18, which is pretrained on ImageNet. It extracts 512 features from the image.

These features go into a shared weight vector, which calculates one ripeness score.

Then, three separate learned biases generate the three probabilities using the sigmoid function.

These probabilities represent whether the grade is greater than zero, one or two.

Because we share the weight vector, the answers remain consistent and cannot contradict each other.

---

### Slide 9 — A Fair, Controlled Comparison

I trained three different models.

The first is a small CNN trained from scratch.

The second is ResNet18 with a Softmax classification head.

The third is ResNet18 with the CORAL ordinal regression head.

To make the comparison fair, I kept the folds, random seeds and training schedule the same.

I used AdamW, cosine learning-rate scheduling and early stopping. For transfer learning, I initially froze the ResNet18 backbone for three epochs and then unfroze it using a lower learning rate.

I also used data augmentation but avoided hue changes because colour is important for ripeness grading.

---

### Slide 10 — Check the Plumbing First

Before starting full training, I performed sanity checks to make sure the models were working correctly.

First, I checked the initial losses. For Softmax, the expected loss was approximately 1.386, and for the ordinal model it was approximately 2.079. The actual values matched these expectations.

Next, I tested whether each model could memorize a single batch of 32 images.

The first ordinal model got stuck at a loss of 0.47. I identified that its thresholds were initialized too close together.

I fixed this by initializing them at 2, 0 and minus 2 and increasing their learning rate ten times.

---

### Slide 11 — What the Numbers Say

Here are the final results from all 45 training runs.

The main metric is MAE, which measures how many grades the model is wrong on average. Lower is better.

The CNN achieved 0.422, ResNet18 with Softmax achieved 0.228, and ResNet18 with ordinal regression achieved 0.214.

The ordinal model also has slightly better accuracy and QWK, with fewer severe errors.

So, transfer learning significantly reduces the error, while ordinal regression provides a small but consistent improvement over Softmax.

However, the difference is not statistically conclusive.

---

### Slide 12 — Findings and Limitations

There are three main findings from this project.

First, transfer learning performs significantly better than training a CNN from scratch.

Second, ordinal regression provides a small improvement over Softmax, particularly in reducing large grading errors.

Third, Grad-CAM shows that the model generally focuses on the fruit for ripe and rotten examples.

There are also limitations. Our dataset is relatively small, some labels contain errors, and most rotten strawberries come from one photo session.

For future improvements, I would collect more diverse images and build an integrated YOLO detection and grading pipeline.

---

### Slide 13 — Thank You

This brings me to the end of my presentation.

Through this project, I developed a complete deep learning pipeline for fruit ripeness grading, from preparing the dataset to training and evaluating models.

The main objective was to explore how ordinal regression can improve predictions when categories have a natural order.

The application is also available as a live browser-based demo, where users can test their own fruit photographs.

Thank you for your time. I am happy to answer any questions.
