import numpy as np
import tensorflow as tf
# np.random.seed(123)
from keras.models import Sequential  # this is a model for a sequential net  keras wil help with'
from keras.layers import Dense, Dropout, Activation, \
    Flatten  # you know what all of these words mean but you should google and confirm
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import pandas as pd
from matplotlib import pyplot as plt

xtrain = pd.read_csv("pricetrain.csv")
test = pd.read_csv("pricetest.csv")
xtrain.fillna(xtrain.mean(), inplace=True)
xtrain.drop(["Alley"], axis=1, inplace=True)
xtrain.drop(["PoolQC"], axis=1, inplace=True)
xtrain.drop(["Fence"], axis=1, inplace=True)
xtrain.drop(["MiscFeature"], axis=1, inplace=True)
xtrain.drop(["PoolArea"], axis=1, inplace=True)
columns = list(xtrain)
for i in columns:
    if xtrain[i].dtypes == 'object':
        xtrain[i] = pd.Categorical(pd.factorize(xtrain[i])[0])
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
for i in columns:
    if xtrain[i].dtypes == 'float32':
        xtrain[i] = le.fit_transform(xtrain[i])
ytrain = xtrain["SalePrice"]
xtrain.drop(["SalePrice"], axis=1, inplace=True)
ytrain = ytrain.values
xtrain = xtrain.values
ytrain = ytrain.astype("float32")
np.savetxt("thecheck.csv", xtrain, fmt='%i', delimiter=',')

model = Sequential(
    [
        Dense(5, activation='relu', input_shape=(75,)),
        # Flatten(),
        Dense(5, activation='relu'),
        Dense(1),
    ])
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(xtrain, ytrain, epochs=100, verbose=1)