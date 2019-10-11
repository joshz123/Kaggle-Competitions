

import plaidml.keras
plaidml.keras.install_backend()
from keras import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from PIL import Image
import pandas as pd
import numpy as np
from ast import literal_eval
import os
raw = pd.read_csv("training.csv")
print(raw.isnull().any().value_counts())
raw.fillna("ffill",inplace=True)
print(raw.isnull().any().value_counts())
print(raw.head())
images = []
for i in range(0,7049):
    img = raw["Image"][i].split(" ")
    img = ['0' if x ==" " else x for x in img]
    images.append(img)

images = np.array(images,dtype='float')
xtrain = images.reshape(-1,96,96)
plt.imshow(xtrain[0].reshape(96,96),cmap='gray')
plt.show()
print(xtrain[0].shape)



#load function taken from
# https://fairyonice.github.io/achieving-top-23-in-kaggles-facial-keypoints-detection-with-keras-tensorflow.html
def load(filepath,test=False, cols=None):
    """
    load test/train data
    cols : a list containing landmark label names.
           If this is specified, only the subset of the landmark labels are
           extracted. for example, cols could be:

          [left_eye_center_x, left_eye_center_y]

    return:
    X: 2-d numpy array (Nsample, Ncol*Nrow)
    y: 2-d numpy array (Nsample, Nlandmarks*2)
       In total there are 15 landmarks.
       As x and y coordinates are recorded, u.shape = (Nsample,30)

    """

    fname = "FTEST" if test else "FTRAIN"
    df = pd.read_csv(filepath)

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:
        df = df[list(cols) + ['Image']]

    myprint = df.count()
    myprint = myprint.reset_index()
    print(myprint)
    ## row with at least one NA columns are removed!
    df = df.dropna()

    X = np.vstack(df['Image'].values) / 255.  # changes valeus between 0 and 1
    X = X.astype(np.float32)

    if not test:  # labels only exists for the training data
        ## standardization of the response
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # y values are between [-1,1]
        X, y = pd.shuffle(X, y, random_state=42)  # shuffle data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y




# model = Sequential()
# model.add(Dense(100,input_dim=X.shape[1],activation="relu"))
# model.add(Dense(30))
#
#
# sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
# model.compile(loss='mean_squared_error', optimizer=sgd)
# hist = model.fit(X, y, nb_epoch=100, validation_split=0.2,verbose=False)











# X_raw = raw["Image"]
# Y_Train = raw[["nose_tip_y", "nose_tip_x"]]
# X_Train =[]
# for j in range(len(X_raw)):
#     print(int(j/len(X_raw)*100), "%")
#     imgarray = [int(i) for i in X_raw[j].split(" ")]
#     temparray1 = []
#     for k in range(96):
#         temparray2 =[]
#         for l in range(96):
#             temparray2.append(imgarray[l + k*96])
#         temparray1.append(np.array(temparray2))
#     X_Train.append(np.array(temparray1))
# X_Train =np.array(X_Train)
#
# # for j in range(len(X_raw)):
# #     print(int(j/len(X_raw)*100), "%")
# #     imgarray = [int(i) for i in X_raw[j].split(" ")]
# #     temparray1 = []
# #     for k in range(96):
# #         for l in range(96):
# #             temparray1.append(imgarray[l + k*96])
# #     X_Test.append(temparray1)
# # df = pd.DataFrame(X_Test)
# # df.to_csv('X_test.csv',index=False)
# X_Train= X_Train/255
# X_Train = X_Train.reshape(X_Train.shape[0], 1, 96, 96).astype('float32')
# X_Train= X_Train/255
# def model():
#     model = Sequential()
#     model.add(Conv2D(32, 5,input_shape=(1, 96,96), activation='relu', data_format = 'channels_first'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.2))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(2))
#     model.compile(loss='mse', optimizer='adam')
#     return model
# model = model()
# history = model.fit(X_Train,Y_Train, validation_split=0.2, epochs=15, verbose=1)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
