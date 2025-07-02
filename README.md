**Problem Description**
To develop an image classifier model which can recognize vegetable item(s) in a photo and identify them for the user 
- The classifier model used images data-set scraped from Google, with 4 labeled classes- Onion, Potato, Tomato and Indian-market. 

**Company background :**
Company E is India’s largest fresh produce supply chain management company, with pioneering work ongoing on solving 
one of the toughest supply chain problems of the World with innovative solutions. 
Their work includes sourcing of fresh produce from farmers and delivering to businesses within 12 hours. 
An important piece of this workflow is development of robust classifier models which can distinguish between images.

**Business goals :** 
The company E wants to build a robust model with multi-class classification functionality for 4 different classes- 
Onions, Tomato, Potato and Indian-market (Noise). 
The deploy-ready solution must have acceptable accuracy levels for the CNN model for the test and training sets.

**Methodology :**

-Import the dataset, verify the count of Images in each train and test folders for each class

-Split the training dataset into train and validation set

-Review the images randomly from each class, verify the size of the image to be consistent across the data-set

-Perform rescaling which will scale the inputs between 0-1 by dividing each value by 255

-Pretrained weights from the popular networks will be good starting point for training the CNN model

-Review the performance of the model- consider batch Normalization, dropouts, Data augmentation and other technique to select best model 

**Tracking Model performance:**

# 97% validation accuracy and 88% test accuracy achieved with mentioned setup

# Closer look at the other performance parameters suggests further fine-tuning required for the CNN model.

# Lower precision and Recall indicate the necessity of adding few more convolution layers, more data augmentation and 
per-class evaluation of metrics to understand the model behavior.

# Mentioned ResNet50 model with Adam optimizer showed promise in the image classification task, however needs further refinement to get to production level metrics

# Confusion matrix reveals model getting confused between tomato and onion- which is understandable- given both are round and red as dominant color

# More convolution layers with potentially Black and white images can help model extract better distinguishing features between tomato and onion



