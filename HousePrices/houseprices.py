# import numpy as np
#
# np.random.seed(123)
# from keras.models import Sequential  # this is a model for a sequential net  keras wil help with'
# from keras.layers import Dense  # you know what all of these words mean but you should google and confirm
# import pandas as pd
#
# xtrain = pd.read_csv("HousePrices/pricetrain.csv")
# test = pd.read_csv("HousePrices/pricetest.csv")
# xtrain.fillna(xtrain.mean(), inplace=True)
# xtrain.drop(["Alley"], axis=1, inplace=True)
# xtrain.drop(["PoolQC"], axis=1, inplace=True)
# xtrain.drop(["Fence"], axis=1, inplace=True)
# xtrain.drop(["MiscFeature"], axis=1, inplace=True)
# xtrain.drop(["PoolArea"], axis=1, inplace=True)
# test.fillna(xtrain.mean(), inplace=True)
# test.drop(["Alley"], axis=1, inplace=True)
# test.drop(["PoolQC"], axis=1, inplace=True)
# test.drop(["Fence"], axis=1, inplace=True)
# test.drop(["MiscFeature"], axis=1, inplace=True)
# test.drop(["PoolArea"], axis=1, inplace=True)
# columns = list(xtrain)
# for i in columns:
#     if xtrain[i].dtypes == 'float32' or xtrain[i].dtypes == 'int32' or xtrain[i].dtypes == 'object':
#         xtrain[i] = pd.Categorical(pd.factorize(xtrain[i])[0])
# from sklearn import preprocessing
#
# le = preprocessing.LabelEncoder()
# for i in columns:
#     if xtrain[i].dtypes == 'float32':
#         xtrain[i] = le.fit_transform(xtrain[i])
# cols = list(test)
# for i in cols:
#     if test[i].dtypes == 'float32' or test[i].dtypes == 'int32' or test[i].dtypes == 'object':
#         test[i] = pd.Categorical(pd.factorize(test[i])[0])
# from sklearn import preprocessing
#
# le = preprocessing.LabelEncoder()
# for i in cols:
#     if test[i].dtypes == 'float32':
#         test[i] = le.fit_transform(xtrain[i])
# ytrain = xtrain["SalePrice"]
#
# xtrain.drop(["SalePrice"], axis=1, inplace=True)
# ytrain = ytrain.values
# xtrain = xtrain.values
# ytrain = ytrain.astype("float32")
# np.savetxt("HousePrices/thecheck.csv", xtrain, fmt='%i', delimiter=',')
#
# model = Sequential(
#     [
#         Dense(100, activation='relu', input_shape=(75,)),
#         # Flatten(),
#         Dense(100, activation='relu'),
#         Dense(100, activation='linear'),
#         Dense(1),
#     ])
# model.compile(loss='mean_absolute_error', optimizer='adam')  # mean absolute vs mean squared error
# model.fit(xtrain, ytrain, epochs=20, verbose=1)
# result = model.predict_classes(test)
# print((result))
# print(ytrain)
#
# # array = np.column_stack((ytrain, result))
#
# np.savetxt("HousePrices/result.csv", result, fmt='%i',delimiter=',')



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

xtrain = pd.read_csv("HousePrices/pricetrain.csv")
test = pd.read_csv("HousePrices/pricetest.csv")
xtrain.fillna(xtrain.mean(), inplace=True)
xtrain.drop(["Alley"], axis=1, inplace=True)
xtrain.drop(["PoolQC"], axis=1, inplace=True)
xtrain.drop(["Fence"], axis=1, inplace=True)
xtrain.drop(["MiscFeature"], axis=1, inplace=True)
xtrain.drop(["PoolArea"], axis=1, inplace=True)
test.fillna(xtrain.mean(), inplace=True)
test.drop(["Alley"], axis=1, inplace=True)
test.drop(["PoolQC"], axis=1, inplace=True)
test.drop(["Fence"], axis=1, inplace=True)
test.drop(["MiscFeature"], axis=1, inplace=True)
test.drop(["PoolArea"], axis=1, inplace=True)
columns = list(xtrain)
for i in columns:
    if xtrain[i].dtypes == 'object':
        xtrain[i] = pd.Categorical(pd.factorize(xtrain[i])[0])
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
for i in columns:
    if xtrain[i].dtypes == 'float32':
        xtrain[i] = le.fit_transform(xtrain[i])
cols = list(test)
for i in cols:
    if test[i].dtypes == 'object':
        test[i] = pd.Categorical(pd.factorize(test[i])[0])



for i in cols:
    if test[i].dtypes == 'float32':
        test[i] = le.fit_transform(test[i])
ytrain = xtrain["SalePrice"]
xtrain.drop(["SalePrice"], axis=1, inplace=True)
ytrain = ytrain.values
xtrain = xtrain.values
ytrain = ytrain.astype("float32")
np.savetxt("HousePrices/thecheck.csv", xtrain, fmt='%i', delimiter=',')
from sklearn.model_selection import train_test_split
#X_train, X_valid, Y_train, Y_valid = train_test_split(xtrain, ytrain, train_size=0.8, test_size= 0.2, random_state=2)
model = Sequential(
    [
        Dense(100, activation='relu', input_shape=(75,)),
        # Flatten(),

        Dense(100, activation='relu'),
        Dense(100, activation='linear'),
        Dropout(rate = 0.1),
        Dense(1),
    ])

print(len(ytrain))
print(len(xtrain))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.fit(xtrain, ytrain, epochs=100, verbose=1)
prediction = model.predict(test)

print(len(prediction))
print(len(test))
from sklearn.metrics import mean_absolute_error
#mae = mean_absolute_error(ytrain,prediction)
array = np.column_stack((test["Id"], prediction))
#
np.savetxt("HousePrices/result.csv", array, fmt='%i',delimiter=',')