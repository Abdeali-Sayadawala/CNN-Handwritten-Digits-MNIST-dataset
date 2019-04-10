import keras
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

# importing and preprocessing the mnist dataset for digit recognition
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# trainig and testing image data preprocessing
x_train = x_train.reshape(60000, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# training and testing label data preprocessing
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

Network = load_model('CNN_Handwritten_Model_2c_2p_hl256_hl128.h5')

y_pred = Network.predict_classes(x_test)
y_pred = np_utils.to_categorical(y_pred)

# see which we predicted correctly and which not
correct_indices = np.nonzero(y_pred == y_test)[0]
incorrect_indices = np.nonzero(y_pred != y_test)[0]
print()
print(len(correct_indices), " classified correctly")
print(len(incorrect_indices), " classified incorrectly")
