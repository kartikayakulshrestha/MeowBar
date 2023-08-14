import tensorflow as tf
#usually tensorflow.keras.models hota hai. but we are writing so 
#because we have the keras file directly installed
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
from keras.models import Sequential

import numpy as nump

import matplotlib.pyplot as plot
#Going to do binary image classification with cnn with tensorflow 

"""Data set loading..."""


#importing images here
L_train= nump.loadtxt("Image Classification CNN Keras Dataset/input.csv" , delimiter=",")
M_train= nump.loadtxt("Image Classification CNN Keras Dataset/labels.csv" , delimiter=",")

L_test=nump.loadtxt("Image Classification CNN Keras Dataset/input_test.csv" , delimiter=",")
M_test=nump.loadtxt("Image Classification CNN Keras Dataset/labels_test.csv" , delimiter=",")

#resizing them in
L_train=nump.reshape(L_train,(len(L_train),100,100,3))
M_train=nump.reshape(M_train,(len(M_train),1))

L_test=nump.reshape(L_test,(len(L_test),100,100,3))
M_test=nump.reshape(M_test,(len(M_test),1))

#we want our pixil to be between to 0 to 1
L_train=L_train/255.0
M_train=M_train/255.0


"""Model"""
model=Sequential([
    Conv2D(32,(3,3),padding="same",activation="relu",input_shape=(100,100,3))
    
])
