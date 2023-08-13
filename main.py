import tensorflow as tf
#usually tensorflow.keras.models hota hai. but we are writing so 
#because we have the keras file directly installed
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
from keras.models import Sequential

import numpy as nump

#Going to do binary image classification with cnn with tensorflow 


#importing images here
L_train= nump.loadtxt("Image Classification CNN Keras Dataset/input.csv" , delimiter=",")
M_train= nump.loadtxt("Image Classification CNN Keras Dataset/labels.csv" , delimiter=",")

L_test=nump.loadtxt("Image Classification CNN Keras Dataset/input_test.csv" , delimiter=",")
M_test=nump.loadtxt("Image Classification CNN Keras Dataset/labels_test.csv" , delimiter=",")

#resizing them in
L_train=nump.reshape(len(L_train),100,100,3)
M_train=nump.reshape(len(M_train),1)

L_test=nump.reshape(len(L_test),100,100,3)
M_test=nump.reshape(len(M_test),1)



print(L_train.shape)