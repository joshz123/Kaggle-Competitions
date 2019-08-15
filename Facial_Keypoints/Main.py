import matplotlib.pyplot as plt
import tensorflow
import keras
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from PIL import Image
import pandas as pd
import numpy as np
from ast import literal_eval

raw = pd.read_csv("training.csv")
X_raw = raw["Image"]
Y_Train = raw[["nose_tip_y", "nose_tip_x"]]
X_Train =[]
for j in range(len(X_raw)):
    print(int(j/len(X_raw)*100), "%")
    imgarray = [int(i) for i in X_raw[j].split(" ")]
    temparray1 = []
    for k in range(96):
        temparray2 =[]
        for l in range(96):
            temparray2.append(imgarray[l + k*96])
        temparray1.append(np.array(temparray2))
    X_Train.append(np.array(temparray1))
X_Train =np.array(X_Train)

# for j in range(len(X_raw)):
#     print(int(j/len(X_raw)*100), "%")
#     imgarray = [int(i) for i in X_raw[j].split(" ")]
#     temparray1 = []
#     for k in range(96):
#         for l in range(96):
#             temparray1.append(imgarray[l + k*96])
#     X_Test.append(temparray1)
# df = pd.DataFrame(X_Test)
# df.to_csv('X_test.csv',index=False)
X_Train= X_Train/255
X_Train = X_Train.reshape(X_Train.shape[0], 1, 96, 96).astype('float32')
X_Train= X_Train/255
def model():
    model = Sequential()
    model.add(Conv2D(32, 5,input_shape=(1, 96,96), activation='relu', data_format = 'channels_first'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2))
    model.compile(loss='mse', optimizer='adam')
    return model
model = model()
history = model.fit(X_Train,Y_Train, validation_split=0.2, epochs=40, verbose=1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
