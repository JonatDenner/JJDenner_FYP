# code adapted from https://www.analyticsvidhya.com/blog/2019/09/step-by-step-deep-learning-tutorial-video-classification-python/

from keras.callbacks import ModelCheckpoint
import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import sys

train = pd.read_csv(sys.path[0] + "/outputs/train_csv/train_new.csv")

# print(train.head())

# creating an empty list
train_image = []

# for loop to read and store frames
for i in tqdm(range(train.shape[0])):
    # loading the image and keeping the target size as (224,224,3)
    img = image.load_img(
        sys.path[0] + "/outputs/train_1/" + train['image'][i], target_size=(224, 224, 3))
    # converting it to array
    img = image.img_to_array(img)
    # normalizing the pixel value
    img = img / 255
    # appending the image to the train_image list
    train_image.append(img)
    print("\033[1;30m" + str(i) + ": Adding " + train['image'][i])

print("---------------------------------------\033[1;37m")
# converting the list to numpy array
X = np.array(train_image)

# shape of the array
# print(X.shape)

# separating the target
y = train['class']

# creating the training and validation set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y)

# creating dummies of target variable for train and validation set
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# creating the base model of pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

# extracting features for training frames
print("\033[1;32mExtracting training frame features. This may take some time...")
X_train = base_model.predict(X_train)
#print(X_train.shape) # (5932, 7, 7, 512)

# extracting features for validation frames
print("\033[1;32mExtracting validation frame features. This may take some time...")
X_test = base_model.predict(X_test)
#print(X_test.shape) # (1483, 7, 7, 512)

# reshaping the training as well as validation frames in single dimension
X_train = X_train.reshape(5932, 7 * 7 * 512)
X_test = X_test.reshape(1483, 7 * 7 * 512)

# defining the model architecture
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(25088,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax')) #set total number of classes as number (can use number of videos in /media/base)

# defining a function to save the weights of best model
print("\033[1;32mSetting model save file...")
mcp_save = ModelCheckpoint(
    'weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')

# compiling the model
print("\033[1;32mCompiling model...")
model.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=['accuracy'])

# training the model
print("\033[1;31mFinding best epoch. This may take some time...")
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[mcp_save], batch_size=128)
