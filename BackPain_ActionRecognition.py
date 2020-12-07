#Back pain action recognition for smart home application
#Author: Vishakha Thakurdwarkar
#CSCE636: Deep Learning


import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing import image
import numpy as np
from skimage.transform import resize
import time

#Starting time of execution
start_time = time.time()

#Extract frames from training video
count = 0
videoFile = "training.mp4"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename ="frame%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
cap.release()
print ("Done!")

#Extract frames from test video
count = 0
videoFile = "test.mp4"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename ="test%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
cap.release()
print ("Done!")

#Read labels from csv files
data = pd.read_csv('mapping.csv')
test = pd.read_csv('testing.csv')

#Read frames for training video
X = []
for img_name in data.Image_ID:
    img = plt.imread('' + img_name)
    X.append(img)
X = np.array(X)
#Read frames for test video
test_image = []
for img_name in test.Image_ID:
    img = plt.imread('' + img_name)
    test_image.append(img)
test_img = np.array(test_image)

#Converts a class vector (integers) to binary class matrix
from keras.utils import np_utils
train_y = np_utils.to_categorical(data.Class)
test_y = np_utils.to_categorical(test.Class)

#Reshaping all images to 224x224x3 as needed by VGG16
image = []
for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224,3)).astype(int)
    image.append(a)
X = np.array(image)
test_image = []
for i in range(0,test_img.shape[0]):
    a = resize(test_img[i], preserve_range=True, output_shape=(224,224)).astype(int)
    test_image.append(a)
test_image = np.array(test_image)

#preprocessing as per the model’s requirement
from keras.applications.vgg16 import preprocess_input
X = preprocess_input(X, mode='tf')
test_image = preprocess_input(test_image, mode='tf')

#validation set to check the performance of the model on unseen images randomly divides images into training and validation set
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, train_y, test_size=0.3, random_state=42)

#import the required libraries to build the model
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
# load the VGG16 pretrained model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
test_image = base_model.predict(test_image)
#reshaping to 1-D
X_train = X_train.reshape(X_train.shape[0], 7*7*512)
X_valid = X_valid.reshape(X_valid.shape[0], 7*7*512)
test_image = test_image.reshape(test_image.shape[0], 7*7*512)
#centering the data
train = X_train/X_train.max()
X_valid = X_valid/X_train.max()
test_image = test_image/test_image.max()

# Building the model
model = Sequential()
model.add(InputLayer((7*7*512,)))                   # input layer
model.add(Dense(units=2048, activation='relu'))     # hidden layer
model.add(Dropout(0.2))
model.add(Dense(units=1024, activation='sigmoid'))  # hidden layer
model.add(Dropout(0.2))
model.add(Dense(units=512, activation='relu'))      # hidden layer
model.add(Dropout(0.5))
model.add(Dense(units=256, activation='relu'))      # hidden layer
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='sigmoid'))  
model.add(Dense(2, activation='softmax'))           # output layer

# Display model summary 
model.summary()

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
class_weights = compute_class_weight('balanced',np.unique(data.Class), data.Class)  # computing weights of different classes

from keras.callbacks import ModelCheckpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]      # model check pointing based on validation loss

# Training the model
model.fit(train, y_train, epochs=15, validation_data=(X_valid, y_valid), class_weight=class_weights, callbacks=callbacks_list)

model.load_weights("weights.best.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
scores = model.evaluate(test_image, test_y)
# Displaying metrics - accuarcy and loss
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Total time taken for execution
print("--- %s seconds ---" % (time.time() - start_time))

# Generating prediction for test video
y = model.predict(test_image)
a, b = y.shape
x = []; xlabel = []; ylabel = []
for i in range(0,a):
    xlabel.append(i)
    ylabel.append(y[i][1])
    x.append([i,y[i][1]])

# Saving the Time-Label information in json file
import json
with open('TimeLabel.json', 'w', encoding='utf-8') as f:
    json.dump("back pain:" + str(x), f, ensure_ascii=False, indent=4)

#Plotting graph of time vs label
import matplotlib.pyplot as plt2
fig = plt2.figure()
plt2.plot(xlabel, ylabel)
plt2.xlabel('Time (s)')
plt2.ylabel('Label (prediction)')
plt2.title('Time-Label Plot')
plt2.show()
fig.savefig('Time-Label Plot.png')