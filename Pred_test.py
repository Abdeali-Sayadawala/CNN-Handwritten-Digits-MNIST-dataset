import keras
from keras.models import load_model
import cv2
import numpy as np

Network = load_model('CNN_Handwritten_Model_2c_2p_hl256_hl128.h5')

img = cv2.imread('cus_eg/2.jpg', 0)
gray = cv2.resize(255-img, (28, 28))

im = np.array(gray)
im = im.astype('float32')
imr = im.reshape(1, 28, 28, 1)
imr /= 255
y_pred = Network.predict_classes(imr)
print(y_pred)
