
# coding: utf-8

# In[1]:


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
get_ipython().run_line_magic('matplotlib', 'inline')


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

images, steering_angle = normalize_by_steering_angle(images, steering_angle)  # normalize by steering angle
print("image shape after processing is", images.shape)

       
# print some random images:
print('printing three random images')
for i in range(3):
    randno = randint(0,len(images))
    plt.imshow(images[randno])
    plt.show()
    

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
                       



# In[ ]:


import keras
print(keras.__version__)

