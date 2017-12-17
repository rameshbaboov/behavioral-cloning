# Behaviorial Cloning Project

## The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect images that provides as a base input to the CNN. The images are captured by driving the vehicle few laps
* Create a pre-processing technique that ensures that the images are prepared to ensure that there are sufficient images other than the ones when car is driving at zero steering angle, so that model can learn on how to avoid going off the raod at turns.
* Create a model and train the Model with the image
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.


## Additional files used
### `drive.py`

This file was modified as my model needs input in a different shape than the one provided by the simulator. so I have modified drive.py to include the same pre-processing routine that is presend in my model.

## Are all the required files submitted
Yes. I have submitted modified drive.py, model.py and model.h5 along with video. In the video, the simulator drives more than one lap without going off the road even at turns.

## Quality of Code
### Is the code functional?
Yes. The code is fully functional. I have captured the output as part of this file to demonstrate that results are produced as required for each of these section

### Is the code usable and readable?
Model.py uses a python generator which enables the model to run simulataneously while the generator generates batch of data. I have created output that depicts the iterations. Comments are included as required

## Model Architecture and Training Strategy

### Has an appropriate model architecture been employed for the task?

I tried the model with a normal Neural network and this didn't run fully. Hence adopted NVIDIA model.  Following is the approach that I had taken:


NVIDIA Model architecture as given in Internet:
![alt text](https://github.com/rameshbaboov/behavioral-cloning/blob/master/model/nVidia_model.png "NVIDIA Model")

My architecture includes the below:

1. Lambda layer to transform the data using function f(x) = x/255 - 0.5. This layer accepts the input shape of (66,200,3)
2. Three convolution layer with activation function. Each of these have depth of 24, 36 and 48 with 2 x 2 stride followed by a drop out
3. two convolution layer layers with output depth of 64
4. Flatten layer
5. Three fully connected layers with depth 120, 80, 10, tanh activation
6. One fully connected layer
7. Loss function of MSE is chose and ADAM optimizer is chosen
8. The architecture uses fit_generator instead of model.fit

### Has an attempt been made to reduce overfitting of the model?

Data has been split into training, validation and testing. Drop out has been added to avoid over fitting

### Have the model parameters been tuned appropriately?

Adam optimizer is used

### Is the training data chosen appropriately?

Training data includes enough number images that covers turning. have also normalized the data to include good number of samples that includes non zero steering angle. Also flipping of images is also used

## Approach

### Logic:

1. There are 21581 images captured from the camera
2. 10238 were filtered out by the normalization function and 11343 images were selected for pre-processing
3. Images are pre-processed to 
    a. crop the images to just show only the road covering the turns
    b. reshape the image to 200,66 as required by the model
    c. color conversion from BGR to YUV as required by the model
4. images are flipped and added to the repository
5. Visualization
    a. randomly three images are displayed to check if the images are loaded correctly
 6. Images are split into Train, test and validation in the ratio of 60%, 20% and 20%
 7. Generators are defined to generate data with a batch size of 50
 8. Data is fed into Fit Generator for training and validation
 
 
 ### Code and Output
 
 '''python
 
 # model change: added lambda layer AND CONvolution2d. added data augmentation and cropping of image
# changing architecture to NVIDIA arch.

#####################################################################################################################
# definition starts
#####################################################################################################################

#pre_process_image - to pre process image
#NVIDIA_ARCH

import csv
import os
import cv2
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from PIL import Image
import numpy as np
%matplotlib inline


from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

''' python
 
