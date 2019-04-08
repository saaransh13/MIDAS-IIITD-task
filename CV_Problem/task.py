import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from sklearn.model_selection import train_test_split
import pickle
import cv2
import pandas as pd
    
with open('train_image.pkl', 'rb') as f:
    train_image = pickle.load(f)

with open('train_label.pkl', 'rb') as f:
    train_label = pickle.load(f)

with open('test_image.pkl', 'rb') as f:
    test_image = pickle.load(f)

for i,x in enumerate(train_image):
    train_image[i] = np.asarray(train_image[i]).reshape(1,28,28)

for i,x in enumerate(test_image):
    test_image[i] = np.asarray(test_image[i]).reshape(1,28,28)
train_image = np.array(train_image)
test_image = np.array(test_image)

train_label = np.array(train_label)

for i,x in enumerate(train_label):
##    if x == 0:
##        train_label[i] = 0
    if x == 2:
        train_label[i] = 1
    elif x == 3:
        train_label[i] = 2
    elif x == 6:
        train_label[i] = 3
## 0 : 0 , 1 : 2, 2 : 3, 3 : 6



train_label = np_utils.to_categorical(train_label, 4)


X_train, X_val, Y_train, Y_val = train_test_split(train_image, train_label, test_size=0.20)

Y_val = np.array(Y_val)


##VGG16 - till layer2_block
##model = Sequential()
##model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(1,28,28)))
##model.add(Conv2D(32, kernel_size = (3, 3), activation='relu'))
##model.add(MaxPooling2D(pool_size=(2,2)))
####model.add(BatchNormalization())
##model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
####model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
####model.add(MaxPooling2D(pool_size=(2,2)))
####model.add(BatchNormalization())
##model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
##model.add(MaxPooling2D(pool_size=(2,2)))
##print(model.output_shape)

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(1,28,28)))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
##model.add(BatchNormalization())
##model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
##model.add(MaxPooling2D(pool_size=(2,2)))
##model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
##model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
##model.add(MaxPooling2D(pool_size=(2,2)))
##model.add(BatchNormalization())
##model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
##model.add(MaxPooling2D(pool_size=(2,2)))
##model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
##model.add(Dropout(0.3))
model.add(Dense(4, activation = 'softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



model.fit(X_train, Y_train, 
          batch_size=32, epochs=35, verbose=1)

score = model.evaluate(X_val, Y_val, verbose=1)

##print(score)

Y_test = model.predict(test_image, verbose=1)

final_y_pred = []
for i,x in enumerate(Y_test):
    final_y_pred.append(np.argmax(Y_test[i]))

for i,x in enumerate(final_y_pred):
    if x == 1:
        final_y_pred[i] = 2
    elif x == 2:
        final_y_pred[i] = 3
    elif x == 3:
        final_y_pred[i] = 6
## 0 : 0 , 1 : 2, 2 : 3, 3 : 6


##Y_pred = np.array(Y_pred)
##
##
##
##final_pred = []
##final_val = []
##
##for i,x in enumerate(Y_pred):
##    final_pred.append(np.argmax(Y_pred[i]))
##
##for i,x in enumerate(Y_val):
##    final_val.append(np.argmax(Y_val[i]))
##
##count = np.sum(np.asarray(final_pred) != np.asarray(final_val))
##
##print('count:',count)




raw_data = {'class':final_y_pred}
df = pd.DataFrame(raw_data, columns = ['class'])
df.index.name = 'index'
df.to_csv('./Saaransh_Pandey.csv')    
