# code adapted from https://www.analyticsvidhya.com/blog/2019/09/step-by-step-deep-learning-tutorial-video-classification-python/

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.applications.vgg16 import VGG16
import cv2
import math
from glob import glob
from scipy import stats as s
import os
import sys
import re

base_model = VGG16(weights='imagenet', include_top=True)

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(1000,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

# loading the trained weights
model.load_weights("weight.hdf5")

# compiling the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

test_files = os.listdir(sys.path[0] + "/media/out/test")

test = pd.DataFrame()
test['video_name'] = test_files
test = test[1:] # removes .gitkeep

test_videos = test['video_name']
#print(test_videos)

# creating the tags
train = pd.read_csv("outputs/train_csv/train_new.csv")
y = train['class']
y = pd.get_dummies(y)

# creating two lists to store predicted and actual tags
predict = []
actual = []

# for loop to extract frames from each test video
for i in tqdm(range(test_videos.shape[0])):
    print("\033[1;36mExtracting test video frames...")
    count = 0
    videoFile = test_videos[i+1]
    cap = cv2.VideoCapture(sys.path[0] + "/media/out/test/" + videoFile)
    success,im = cap.read()
    files = glob(sys.path[0] + '/temp/*')
    for f in files:
        os.remove(f)

    while success:
        if count < 10:
            cv2.imwrite(sys.path[0] + "/temp/" + "frame%d.jpg" % count, im)
        else:
            cv2.imwrite(sys.path[0] + "/temp/" + "frame%d.jpg" % count, im)
        success,im = cap.read()
        count += 1
    cap.release()

    # reading all the frames from temp folder
    images = glob("temp/*.jpg")
    images = sorted(images)
    #print(images)

    prediction_images = []
    for i in images:
        img = image.load_img(i, target_size=(224,224,3))
        img = image.img_to_array(img)
        img = img/255
        prediction_images.append(img)

    print("\033[1;36mPredicting...")
    # converting all the frames for a test video into numpy array
    prediction_images = np.array(prediction_images)
    # extracting features using pre-trained model
    prediction_images = base_model.predict(prediction_images)
    # converting features in one dimensional array
    prediction_images = prediction_images.reshape(prediction_images.shape[0], 1000)
    # predicting tags for each array
    predict_x=model.predict(prediction_images)
    classes_x=np.argmax(predict_x,axis=1)
    # appending the mode of predictions in predict list to assign the tag to the video
    predict.append(y.columns.values[s.mode(classes_x)[0][0]])
    # appending the actual tag of the video
    actual.append(videoFile.split('_')[0])

# checking the accuracy of the predicted tags
from sklearn.metrics import accuracy_score
predictions = list(zip(predict, actual))
print("\033[1;36m----------------------------------")
for x in predictions:
    p = x[0]
    a = x[1]
    p = re.sub(r"(\w)([A-Z])", r"\1 \2", p)
    a = re.sub(r"(\w)([A-Z])", r"\1 \2", a)

    if p == a:
        print("\033[1;32mPrediction matched result: " + p)
    else:
        print("\033[1;31mPrediction: " + p + "\tActual: " + a)

score = accuracy_score(predict, actual)*100
print("\033[1;36m----------------------------------\nAccuracy:")
print(score)
