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
 


```python
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


def normalize_by_steering_angle(images, steering_angle):
    
    print('chart before steeting angle normalization')
    no_of_bins = 20
    average_samples_per_bin = len(steering_angle)/no_of_bins
    hist,bins = np.histogram(steering_angle,no_of_bins)
    width = 0.5 * (bins[1]- bins[0])
    center = (bins[:-1] + bins[1:])  / 2
    plt.bar(center,hist,align='center',width= width)
    plt.plot((np.min(steering_angle), np.max(steering_angle)), (average_samples_per_bin,average_samples_per_bin), 'k-')
    plt.show()
    
    
    keep_prob = []
    
    delete_list = []
    keep_factor = 0.3   # tune this
    threshold = average_samples_per_bin * keep_factor
    for index in range(no_of_bins):
        if hist[index] < threshold:
            keep_prob.append(1.)   # keep entire data in the bin
        else:
             keep_prob.append(1. * threshold/hist[index])
    for index in range(len(steering_angle)):
        for index1 in range(no_of_bins):
            if (steering_angle[index] > bins[index1]) and (steering_angle[index] <= bins[index1+1]):
                # delete from X and y with probability 1 - keep_probs[j]
                if np.random.rand() > keep_prob[index1]:
                     delete_list.append(index)
    
    images = np.delete(images,delete_list,axis=0)
    steering_angle = np.delete(steering_angle,delete_list)
             
    print('chart after steeting angle normalization')
             
    hist,bins = np.histogram(steering_angle,no_of_bins)
    width = 0.5 * (bins[1]- bins[0])
    center = (bins[:-1] + bins[1:])  / 2
    plt.bar(center,hist,align='center',width= width)
    plt.plot((np.min(steering_angle), np.max(steering_angle)), (average_samples_per_bin,average_samples_per_bin), 'k-')
    plt.show()
             
    return images,steering_angle
    

           
def pre_process_image(image):
    # original shape: 160x320x3
    processed_image = image[40:160,:,:]
    # cropped to 40x320x3
    # apply subtle blur
    #processed_image = cv2.GaussianBlur(processed_image,(3,3),0)
    # reshape to fit the size for NVIDIA
    
    # debugging######################################
    #plt.imshow(image)
    #print("original image shape", image.shape)
    #plt.imshow(processed_image)
    #print("processed image shape", processed_image.shape)
    #destRGB = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
    #plt.imshow(destRGB)
    #print("BGR image shape", destRGB.shape)
    # debugging######################################
    
    
    
    processed_image = cv2. resize(processed_image,(200,66),interpolation = cv2.INTER_AREA)
     # convert to YUV color space as per NVIDIA requirement
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2YUV)
    return processed_image

def generate_data(images,steering_angle,batch_size):
   
    print("generate_data:images",len(images))
    print("generate_data:steering_angle",len(steering_angle))
    print("generate_data:images shape",images.shape)
    print("generate_data:steering_angle shape",steering_angle.shape) 
    print('printing from generate_data')
    iter = 1
    images, steering_angle = shuffle(images,steering_angle)
    X,Y = ([],[])
    while True:
        for index in range(len(steering_angle)):
            #debugging##############################
            #print(index)
            #debugging##############################
            X.append(images[index])
            Y.append(steering_angle[index])
            if len(X) == batch_size:
                Xarray = np.asarray(X)
                Yarray = np.asarray(Y)
                print("statement before yield")
                print("array x length - ", len(Xarray))
                print("array x length - ", len(Yarray))
                print("array x shape", Xarray.shape)
                print("array x shape", Yarray.shape)
                print("sending data from generator - iteration", iter)
                iter += 1
                yield (Xarray,Yarray)
                print("statement after yield")
                X, Y = ([],[])
                images, steering_angle = shuffle(images,steering_angle)
                       
                       
 

 #####################################################################################################################   
#Main program starts
#####################################################################################################################


# read files and check if file exists
from pathlib import Path
             

lines = []
with open('./data/driving_log.csv') as csvfile:
    next(csvfile) # if header is available
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    print("initial no of one camera images", len(lines))
        
        
images = []
steering_angle = []


for line in lines:
    
# load images and start preprocessing of image
      
    
    source_path = line[0]
    pth, filename = os.path.split(source_path)
    # debugging######################################
    #print("path is ", pth)
    #print("filename is ", filename)
    current_path = './data/img/' + filename
    #print(current_path)
             
# check if the image exists in disk
    myfile = Path(current_path)
    # debugging######################################
    #print("source path is ", pth)
    #print("filename is ", filename)
    # debugging######################################
    if myfile.is_file():         # file exists
        image = cv2.imread(current_path)
        # debugging######################################
        #plt.imshow(image)
        # debugging######################################
        image = pre_process_image(image)
        # flip the images and add flipped images to the original image list
    
        images.append(image)
        angle = float(line[3])
        steering_angle.append(angle)
        # debugging######################################
        #print("angle is ",angle)
        #print('size of images before flipping', len(images))
        #print('size of steering angle before flipping', len(steering_angle))
        # debugging######################################

        image_flipped = np.fliplr(image)
        images.append(image_flipped)
        steering_angle.append(-1. * angle)
        # debugging######################################
        #print('size of images before flipping', len(images))
        #print('size of steering angle before flipping', len(steering_angle))
        # debugging######################################
        print("image shape after processing is", images.shape)

images, steering_angle = normalize_by_steering_angle(images, steering_angle)  # normalize by steering angle

```

   initial no of one camera images 21581
   chart before steeting angle normalization
    
   ![png](https://github.com/rameshbaboov/behavioral-cloning/blob/master/model/output_0_2.png)

   chart after steeting angle normalization
    
   ![png](https://github.com/rameshbaboov/behavioral-cloning/blob/master/model/output_0_4.png)


   image shape after processing is (11343, 66, 200, 3)


```python

      
# print some random images:
print('printing three random images')
for i in range(3):
    randno = randint(0,len(images))
    plt.imshow(images[randno])
    plt.show()
    
```
  

printing three random images 
    


![png](https://github.com/rameshbaboov/behavioral-cloning/blob/master/model/output_0_6.png)



![png](https://github.com/rameshbaboov/behavioral-cloning/blob/master/model/output_0_7.png)



![png](https://github.com/rameshbaboov/behavioral-cloning/blob/master/model/output_0_8.png)


```python


# split the data between training and testing
print("splitting the data")
X_train, X_test, Y_train, Y_test     = train_test_split(images, steering_angle, test_size=0.2, random_state=1)
X_train, X_valid, Y_train, Y_valid   = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
print("length of train data",len(X_train))
print("length of test data",len(X_test))
print("length of validation data",len(X_valid))
print("training dimensions", X_train.shape, Y_train.shape)
print("test dimensions", X_test.shape, Y_test.shape)
print("starting generator")                       
print(X_train[0])



                       
# define generators
train_gen = generate_data(X_train,Y_train, batch_size =50)
val_gen = generate_data(X_valid,Y_valid,   batch_size =50)
test_gen = generate_data(X_test,Y_test,    batch_size =50)
 

# Train the network
# train the model   
                    

model = Sequential()

# Normalize
# compatibility issue
#model.add(Lambda(lambda x: (x /255.0) - 0.5), input_shape = (66,200,3))

model.add(Lambda(lambda x: (x /255.0) - 0.5, input_shape= (66,200,3)))

# add three convolution layers (output depth - 24, 36, and 48; 2x2 stride)
model.add(Convolution2D(24,5,5,subsample =(2,2), activation='relu'))
model.add(Convolution2D(36,5,5,subsample =(2,2), activation='relu'))
model.add(Convolution2D(48,5,5,subsample =(2,2), activation='relu'))

# add dropout
model.add(Dropout(0.5))

 # add two convolution layers (output depth - 64, 64)
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))

# add a flatten layer
model.add(Flatten())

# three fully connected layers (depth 120, 80, 10, tanh activation (and dropouts)
model.add(Dense(120, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(10, activation='relu'))

# fully connected output layer
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')
 
                      
checkpoint = ModelCheckpoint('model_{epoch:02d}_{val_loss:.2f}.h5')

history = model.fit_generator(train_gen, steps_per_epoch=(len(X_train)/50), validation_data = val_gen,nb_epoch=5, verbose=2, nb_val_samples=len(X_valid), callbacks=[checkpoint])
 
print(model.summary())
                       
# save the model

model.save_weights('./model_wt_5.h5')
model.save('model_5.h5')
json_string = model.to_json()
with open('./model.json', 'w') as f:
     f.write(json_string)
                       
```

   generate_data:images 7259
   generate_data:steering_angle 7259
   generate_data:images shape (7259, 66, 200, 3)
   generate_data:steering_angle shape (7259,)
   printing from generate_data
   Epoch 1/5

   sending data from generator - iteration 9079
   statement after yield
   statement before yield
   array x length -  50
   array x length -  50
   array x shape (50, 66, 200, 3)
   array x shape (50,)
   sending data from generator - iteration 9080
    - 64s - loss: 0.0537 - val_loss: 0.0541
   _________________________________________________________________
   Layer (type)                 Output Shape              Param #   
   =================================================================
   lambda_1 (Lambda)            (None, 66, 200, 3)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824      
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636     
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 5, 22, 48)         0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712     
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928     
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 1152)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 120)               138360    
    _________________________________________________________________
    dense_2 (Dense)              (None, 80)                9680      
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                810       
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 11        
   =================================================================
   Total params: 280,209
   Trainable params: 280,209
   Non-trainable params: 0
    _________________________________________________________________
    None
 


```
