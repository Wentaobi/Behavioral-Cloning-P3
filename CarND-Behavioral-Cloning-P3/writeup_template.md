# **Behavioral Cloning** 

## Behavioral Cloning: Navigating a Car in a Simulator

### Overview

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/conv_architecture.png "Model Visualization"
[image2]: ./examples/center.png "Grayscaling"
[image3]: ./examples/left.png "Recovery Image"
[image4]: ./examples/center.png "Recovery Image"
[image5]: ./examples/right.png "Recovery Image"
[image6]: ./examples/flipped.png "Normal Image"
[image7]: ./examples/cropped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.json containing a trained convolution neural network, than convert to h5 format
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

## Implementation

### My model architecture ###

1, Convert/Normalize/Resize raw RGB image 

2, Feed the image to the Network

3, Convolution Layer 1 (Kernel 5*5) + Relu function activation +  Max-pooling layer (2x2)

4, Convolution Layer 2 (Kernel 5*5) + Relu function activation +  Max-pooling layer (2x2)

5, Convolution Layer 3 (Kernel 5*5) + Relu function activation +  Max-pooling layer (2x2)

6, Convolution Layer 4 (Kernel 3*3) + Relu function activation +  Max-pooling layer (2x2)

7, Convolution Layer 5 (Kernel 3*3) + Relu function activation +  Max-pooling layer (2x2)

8, Fully connected layer 1 (FCN) Flatten to 1164

9, Fully connected layer 2 (FCN) Flatten to 100

10, Fully connected layer 3 (FCN) Flatten to 50

11, Fully connected layer 4 (FCN) Flatten to 10

12, Linearize function: 1 output Agent car steering torque

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The images captured by the simulator come with a lot of details which do not directly help model building process. In addition to that extra space occupied by these details required additional processing power. Hence, we remove 35 percent of the original image from the top and 10 percent. This can make our data more relevant to our output, and we normalize our image pixrl to -1 ~ 1.Also, the model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, loss = "mse", mean square error. so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. The dataset consists of 24108+ images which contains left side view, center side camera view and right side camera view. The training track contains a lot of shallow turns and straight road segments. Hence, the majority of the recorded steering angles are zeros. Therefore, preprocessing images and respective steering angles are necessary in order to generalize the training model for unseen tracks such as our validation track. Also, I add bias for left and right which are +/- 0.3

### Model Architecture and Training Strategy

#### 1. Solution Design Approach and 2. Final Model Architecture

The overall strategy for deriving a model architecture was to convert current vehicle camera detected images (left view, center view, right view) to the final steering torque. This torque will let vehicle move from lateral side, and we are assuming longtiunal control was good from simulator itself.

1, Convert/Normalize/Resize raw RGB image 

2, Feed the image to the Network

3, Convolution Layer 1 (Kernel 5*5) + Relu function activation +  Max-pooling layer (2x2)

4, Convolution Layer 2 (Kernel 5*5) + Relu function activation +  Max-pooling layer (2x2)

5, Convolution Layer 3 (Kernel 5*5) + Relu function activation +  Max-pooling layer (2x2)

6, Convolution Layer 4 (Kernel 3*3) + Relu function activation +  Max-pooling layer (2x2)

7, Convolution Layer 5 (Kernel 3*3) + Relu function activation +  Max-pooling layer (2x2)

8, Fully connected layer 1 (FCN) Flatten to 1164

9, Fully connected layer 2 (FCN) Flatten to 100

10, Fully connected layer 3 (FCN) Flatten to 50

11, Fully connected layer 4 (FCN) Flatten to 10

12, Linearize function: 1 output Agent car steering torque

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to turn when the vehicle is not in the center line. These images show what a recovery looks like starting from left camera side, center camera side and right camera view side. 

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help us get more data and let vehicle know each side environment when vehicle drive on road, for example, if vehicle see left curve it know it will turn left, after this operation, the car will also know it will turn right whe right curve comes. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Even after cropping and resizing training images (with all augmented images), training dataset was very large and it could not fit into the main memory. Hence, we used `fit_generator` API of the Keras library for training our model.

We created two generators namely:
* `train_gen = helper.generate_next_batch()`
* `validation_gen = helper.generate_next_batch()` 

Batch size of both `train_gen` and `validation_gen` was 64. We used 20032 images per training epoch. It is to be noted that these images are generated on the fly using the document processing pipeline described above. In addition to that, we used 6400 images (also generated on the fly) for validation. We used `Adam` optimizer with `1e-4` learning rate. After long time testing, `8` works well on both training and validation tracks. 

