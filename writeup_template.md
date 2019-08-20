# **Behavioral Cloning** 

## Writeup Template

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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model looks like follows: 

Image normalization
Convolution: 5x5, filter: 24, strides: 2x2, activation: RELU
Convolution: 5x5, filter: 36, strides: 2x2, activation: RELU
Drop out (0.5)
Convolution: 5x5, filter: 48, strides: 2x2, activation: RELU
Drop out (0.5)
Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
Drop out (0.5)
Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
Drop out (0.5)
Dense: neurons: 100
Dense: neurons: 50
Dense: neurons: 10
Dense: neurons: 1 (output)

The model includes RELU layers to introduce nonlinearity (code line 46), and the data is normalized in the model using a Keras lambda layer (code line 44). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 48). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. To reduce the overfitting I included several Dropout layers after every convlution. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 63).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the architecture published by the autonomous vehicle team at NVIDIA, which is the network they use for training a real car to drive autonomously, I thought this model might be appropriate because is what the course is using as an example and it's functioanllity is already proved. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. This happened because I was running a low number of epochs and noticed that the squared error was not changing to much. After so many attempts I realized that by increasing the number of epochs by 25, the validation loss was not increasing and the model managed to drive a loop for the first time. 

To combat the overfitting, I modified the model so that the dropouts were included after every convolution rather than just one dropout layer at the end. This might seem to extreme but it actually worked well. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track for the first 15 runs of my model, then it was when I decided to increase the number of epochs and this reduced the instances of when the vehicle was baiased to the edges of the track. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 43-58) consisted of a convolution neural network with the following layers and layer sizes ...

Image normalization
Convolution: 5x5, filter: 24, strides: 2x2, activation: RELU
Convolution: 5x5, filter: 36, strides: 2x2, activation: RELU
Drop out (0.5)
Convolution: 5x5, filter: 48, strides: 2x2, activation: RELU
Drop out (0.5)
Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
Drop out (0.5)
Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
Drop out (0.5)
Dense: neurons: 100
Dense: neurons: 50
Dense: neurons: 10
Dense: neurons: 1 (output)

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the data provided in the workspace as I was not able to record data due to this issue "ln: failed to create symbolic link 'CarND-Behavioral-Cloning-P3/data/data': File exists" I tried googling about it and also sent feedback but I didn't get any reponse so since I am behind schedule I just decided to use the data provided from 2016. You can refer to the images in the folder data 

![alt text][image2]

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would prevent the vehicle from steering to the left more as a consequence of the way the track is designed. Even when it was supposed to go straight it was noticable that the steering was biased to the left. So by flipping the images the model can generalize better and avoid being biased when steering. 

![alt text][image6]
![alt text][image7]

Etc ....

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 25. I used an adam optimizer so that manually training the learning rate wasn't necessary.
