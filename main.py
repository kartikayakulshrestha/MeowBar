import tensorflow as tf
#usually tensorflow.keras.models hota hai. but we are writing so 
#because we have the keras file directly installed
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
from keras.models import Sequential

import numpy as nump

#Going to do binary image classification with cnn with tensorflow 
print(tf.__version__,nump.__version__)

