import numpy as np
import tensorflow as tf
#np.random.seed(123)
from keras.models import Sequential  # this is a model for a sequential net  keras wil help with'

from keras.layers import Dense, Dropout, Activation, Flatten  # you know what all of these words mean but you should google and confirm

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils

import pandas as pd
from matplotlib import pyplot as plt



train  = pd.read_csv(R"C:\Users\joshz\OneDrive - University of Waterloo\Desktop\digittrain.csv")
test = pd.read_csv(R"C:\Users\joshz\OneDrive - University of Waterloo\Desktop\test.csv")

ytrain = train["label"]

train.drop(['label'], axis=1, inplace=True)
xtrain = train.values
xtest  = test.values
ytrain = ytrain.values
xtrain = xtrain.astype("float32")
xtest = xtest.astype("float32")


ytrain = ytrain.astype("float32")

ytrain = np_utils.to_categorical(ytrain, 10)
xtest/=255
xtrain/=255
xtest = xtest.reshape((28000,28,28,1))
xtrain = xtrain.reshape((42000,28,28,1))
model = Sequential()

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28, 28, 1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(xtrain, ytrain, batch_size=32, epochs=1)

result = model.predict_classes(xtest,batch_size=32,verbose=1)

x = [x for x in range(1,len(xtest)+1)]
a = np.array(x)
print(a.shape)
print(result.shape)
array = np.column_stack((a,result))

np.savetxt("resultdig.csv",array, fmt = '%i', delimiter=',')

print("done")
