import keras 
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import pickle 

# Importing MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing the MNIST dataset 
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Making categories of the the output data
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Model
Model = Sequential()
Model.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation='relu'))
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Conv2D(64, (3, 3), activation='relu'))
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Dropout(0.2))
Model.add(Flatten())
Model.add(Dense(256, activation='relu'))
Model.add(Dense(128, activation='relu'))
Model.add(Dense(10, activation='softmax'))

# Compiling
Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
Model.fit(x_train, y_train, batch_size=128, epochs = 15)

# Save
Model.save('CNN_Handwritten_Model_2c_2p_hl256_hl1dfcd28.h5')
