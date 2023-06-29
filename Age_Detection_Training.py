import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, AveragePooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import tensorflow as tf

from keras.layers import Dense, Activation
from keras import Sequential
from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)


all_images = os.listdir('combined_faces/')
ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']

X = []
y = []

l = len(all_images)

for a in range(l):
    X.append(cv2.imread(f'combined_faces/{all_images[a]}', 0))
    age = int(all_images[a].split('_')[0])

    if age>=1 and age<=2:
        y.append(0)
    elif age>=3 and age<=9:
        y.append(1)
    elif age>=10 and age<=20:
        y.append(2)
    elif age>=21 and age<=27:
        y.append(3)
    elif age>=28 and age<=45:
        y.append(4)
    elif age>=46 and age<=65:
        y.append(5)
    elif age>=66 and age<=116:
        y.append(6)
    print(str(a)+'/'+str(l))

np.savez_compressed('compressed_image_data.npz', x=X, y=y)

loaded = np.load('compressed_image_data.npz')

X = loaded['x']
Y = loaded['y']


plt.imshow(X[0], cmap='gray')


Y = np_utils.to_categorical(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

X_train = np.array(X_train).reshape(-1, 200, 200, 1)
X_test = np.array(X_test).reshape(-1, 200, 200, 1)


IMG_HEIGHT = 200
IMG_WIDTH = 200
IMG_SIZE = (IMG_HEIGHT,IMG_WIDTH)
batch_size = 128
epochs = 60


train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   height_shift_range=0.1,
                                   width_shift_range=0.1,
                                   rotation_range=15)

test_datagen = ImageDataGenerator(rescale=1./255)


train_data = train_datagen.flow(X_train, Y_train, batch_size)
test_data = test_datagen.flow(X_test, Y_test, batch_size)


final_cnn = Sequential()

final_cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(200, 200, 1)))
final_cnn.add(AveragePooling2D(pool_size=(2,2)))

final_cnn.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
final_cnn.add(AveragePooling2D(pool_size=(2,2)))

final_cnn.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
final_cnn.add(AveragePooling2D(pool_size=(2,2)))

final_cnn.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
final_cnn.add(AveragePooling2D(pool_size=(2,2)))

final_cnn.add(GlobalAveragePooling2D())

final_cnn.add(Dense(132, activation='relu'))

final_cnn.add(Dense(7, activation='softmax'))

final_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


checkpoint = ModelCheckpoint(filepath='final_cnn_model_checkpoint.h5',
                             monitor='val_accuracy',
                             save_best_only=True,
                             save_weights_only=False,
                             verbose=1)


history = final_cnn.fit(train_data,
                        batch_size=batch_size,
                        validation_data=test_data,
                        epochs=epochs,
                        callbacks=[checkpoint],
                        shuffle=False)


plotting_data_dict = history.history

plt.figure(figsize=(12, 8))

test_loss = plotting_data_dict['val_loss']
training_loss = plotting_data_dict['loss']
test_accuracy = plotting_data_dict['val_accuracy']
training_accuracy = plotting_data_dict['accuracy']

epochs = range(1, len(test_loss) + 1)

plt.subplot(121)
plt.plot(epochs, test_loss, marker='X', label='test_loss')
plt.plot(epochs, training_loss, marker='X', label='training_loss')

plt.subplot(122)
plt.plot(epochs, test_accuracy, marker='X', label='test_accuracy')
plt.plot(epochs, training_accuracy, marker='X', label='training_accuracy')
plt.legend()

plt.savefig('training.png')