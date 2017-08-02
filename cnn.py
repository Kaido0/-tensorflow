# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pdb

import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
import keras.utils.np_utils as kutils
import os
from keras.models import load_model
import matplotlib.pyplot as plt




nb_epoch = 1 # Change
#  to 100
batch_size = 128
img_rows, img_cols = 28, 28

nb_filters_1 = 32 # 64
nb_filters_2 = 64 # 128
nb_filters_3 = 128 # 256
nb_conv = 3

path=os.getcwd()
train = np.loadtxt(path + "/train.txt")
val = np.loadtxt(path + "/val.txt")

y_train=train[:,0]
x_train=np.array(train[:,1:])

y_val=val[:,0]
x_val=np.array(val[:,1:])


valY = kutils.to_categorical(y_val)
valX = x_val.reshape(x_val.shape[0], 28, 28, 1)
valX = valX.astype(float)

trainX = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
trainX = trainX.astype(float)
print trainX.shape
trainY = kutils.to_categorical(y_train)
nb_classes = trainY.shape[1]
print nb_classes

cnn = models.Sequential()
cnn.add(conv.Conv2D(nb_filters_1,(nb_conv, nb_conv),padding='same',activation="relu", input_shape=(28, 28, 1)))
cnn.add(conv.Conv2D(nb_filters_1,( nb_conv, nb_conv), padding='same' ,activation="relu"))
cnn.add(conv.Conv2D(nb_filters_1, (nb_conv, nb_conv), padding='same',activation="relu"))
cnn.add(conv.MaxPooling2D(strides=(2,2)))

cnn.add(conv.Conv2D(nb_filters_2, (nb_conv, nb_conv), padding='same',activation="relu"))
cnn.add(conv.Conv2D(nb_filters_2, (nb_conv, nb_conv), padding='same',activation="relu"))
cnn.add(conv.MaxPooling2D(strides=(2,2)))

#cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
#cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
#cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
#cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
#cnn.add(conv.MaxPooling2D(strides=(2,2)))

cnn.add(core.Flatten())
cnn.add(core.Dropout(0.2))
cnn.add(core.Dense(128, activation="relu")) # 4096
cnn.add(core.Dense(nb_classes, activation="softmax"))

cnn.summary()
cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history=cnn.fit(trainX, trainY, batch_size=batch_size, epochs=nb_epoch, validation_data=(valX,valY),verbose=2)

history_dict = history.history
print history_dict.keys()

#model = load_model('my_model.h5')


final_loss, final_acc = cnn.evaluate(valX, valY, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
# summarize history for accuracy

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
cnn.save('my_model.h5')

#def predict():
model = load_model('my_model.h5')
yPre = cnn.predict_classes(valX) #这个位置写不带标签的测试集，yPred就是结果
print yPre