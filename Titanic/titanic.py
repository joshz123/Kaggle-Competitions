import numpy as np

np.random.seed(123)
from keras.models import Sequential  # this is a model for a sequential net  keras wil help with'

from keras.layers import Dense  # you know what all of these words mean but you should google and confirm

import pandas as pd

xtest = pd.read_csv('titanictest.csv')
xtrain = pd.read_csv('titanictrain.csv')
xtrain.drop(['Ticket'], axis=1, inplace=True)
xtrain.drop(['Cabin'], axis=1, inplace=True)
xtrain.drop(['Embarked'], axis=1, inplace=True)
xtrain.drop(['PassengerId'], axis=1, inplace=True)
xtrain.drop(['Name'], axis=1, inplace=True)
xtest.drop(['Ticket'], axis=1, inplace=True)
xtest.drop(['Cabin'], axis=1, inplace=True)
ylabels = xtest['PassengerId']
xtest.drop(['PassengerId'], axis=1, inplace=True)
xtest.drop(['Name'], axis=1, inplace=True)
xtest.drop(['Embarked'], axis=1, inplace=True)

xtrain.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

from sklearn import preprocessing


# print(xtrain.Sex)


def normalize(dataset):
    dataNorm = ((dataset - dataset.min()) / (dataset.max() - dataset.min())) * 20
    dataNorm["Sex"] = dataset["Sex"]
    return dataNorm


xtrain.Sex = pd.Categorical(pd.factorize(xtrain.Sex)[0])
print(xtrain.head(5))
ytrain = xtrain['Survived'].values
xtrain.drop(['Survived'], axis=1, inplace=True)
x = xtrain.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
norm_xtrain = pd.DataFrame(x_scaled)

norm_xtrain = norm_xtrain.values

xtest.Sex = pd.Categorical(pd.factorize(xtest.Sex)[0])

xt = xtest.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaledt = min_max_scaler.fit_transform(xt)
norm_xtest = pd.DataFrame(x_scaledt)

norm_xtest = norm_xtest.values

model = Sequential()
model.add(Dense(units=64, activation='sigmoid', ))
model.add(Dense(units=64, activation='relu', ))

model.add(Dense(units=2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(norm_xtrain, ytrain, batch_size=1, epochs=15, verbose=1)

result = model.predict_classes(norm_xtest, batch_size=1)
# print(len(result))
# print(len(ylabels))
print("hello")
print(result)
print((ylabels))

array = np.column_stack((ylabels, result))

np.savetxt("result.csv", array, fmt='%i', delimiter=',')
