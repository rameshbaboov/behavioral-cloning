# Behaviorial Cloning Project

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect images that provides as a base input to the CNN. The images are captured by driving the vehicle few laps
* Create a pre-processing technique that ensures that the images are prepared to ensure that there are sufficient images other than the ones when car is driving at zero steering angle, so that model can learn on how to avoid going off the raod at turns.
* Create a model and train the Model with the image
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.


### Additional files used
### `drive.py`

This file was modified as my model needs input in a different shape than the one provided by the simulator. so I have modified drive.py to include the same pre-processing routine that is presend in my model.

### Are all the required files submitted
Yes. I have submitted modified drive.py, model.py and model.h5 along with video
