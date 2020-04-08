import cv2 as cv2
import numpy as np
from os import listdir

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


''' importing data into X and Y'''

directory = 'brain_tumor_dataset/'

data_yes = directory +'yes/'
data_no = directory +'no/'

dataset = [data_yes,data_no]


X = []
Y = []
dim = (32,32)
for i in dataset:
    for file in listdir(i):
        img = cv2.imread(i+file)
        img = cv2.resize(img,dim,interpolation=cv2.INTER_CUBIC)
        img = img/255.
        X.append(img)
        
        if i[-4:] =='yes/':
            Y.append([1])
        else:
            Y.append([0])
X = np.array(X)
Y = np.array(Y)

''' shuffle data'''

X,Y = shuffle(X,Y)

''' divide into train and test sets'''

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

''' building the model'''

# input image dimensions
img_rows, img_cols = 32, 32

''' loading the data, already divided into train and test'''

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)

else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)
    
num_classes = 2

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

''' architecture of the convolutional network'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),strides = (1,1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2),strides = (2,2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

''' define loss and optimizer'''
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

''' training'''
model.fit(x_train, y_train,
          batch_size=32,
          epochs=5,
          verbose=1,
          validation_data=(x_test, y_test))

''' accuracy'''
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])