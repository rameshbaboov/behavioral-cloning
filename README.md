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



Inline-style: 
![alt text](https://github.com/rameshbaboov/behavioral-cloning/blob/master/model/nVidia_model.png "NVIDIA Model")
