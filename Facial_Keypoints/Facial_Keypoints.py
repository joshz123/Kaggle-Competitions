import plaidml.keras
plaidml.keras.install_backend()
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Dropout
import pandas as pd
import numpy as np
from sklearn import preprocessing as sk
raw = pd.read_csv("training.csv")
raw.fillna(method = "ffill",inplace=True)
raw.isnull().any().value_counts()
images = []
for i in range(0,7049):
    img = raw["Image"][i].split(" ")
    img = ['0' if x ==" " else x for x in img]
    images.append(img)
images = np.array(images,dtype='float')

xtrain = images.reshape(-1,96,96)
scaler = sk.MinMaxScaler()
scaled_images = scaler.fit_transform(images)
ytrain = raw.drop(["Image"],axis=1)
ytrain_scaled = scaler.fit_transform(ytrain)

model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(96, 96, 1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(30, activation='softmax'))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

scaled_images=scaled_images.reshape(7049,96,96,1)
model.fit(scaled_images, ytrain, batch_size=32, epochs=20,verbose=1,validation_split=0.2)

