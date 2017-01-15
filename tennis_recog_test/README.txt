README

This package containes two simple programs. The purpose of tennis_train.cpp is train a neural network to recognize if there is a tennis racket in the image or not.
The tennis_recog.cpp is a code to load a preexisting network to recognize if there is a tennis racket in the image, then filename of which is provided by the user.
For compilation, clone, create a build folder in tennis_recog_test flder, then you navigate to the build folder, then "cmake .." and then "make". 

There might be some issues with compilation and cmake. Will make this general in the next version.  

As this is a first attempt, the accuracy doesn't seem to be great during the testing 
process. Traning on much higher number of images or different images will be done in the next version. 

The folder of training_images left blank for now in the git repo.
training_labels.txt contains binary labels for around 50 images. This can be modified accordingly

Convolution neural network created using tiny-dnn library. 

 
