# code adapted from https://www.analyticsvidhya.com/blog/2019/09/step-by-step-deep-learning-tutorial-video-classification-python/

import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
import os
import sys

train_files = os.listdir(sys.path[0] + "/media/out/train")

# set up train dataframe
train = pd.DataFrame()
train['video_name'] = train_files
train = train[1:] # removes .gitignore


# tag creation for training videos
train_video_tag = []
for i in range(train.shape[0]):
    train_video_tag.append(train['video_name'][i].split('_')[0])

train['tag'] = train_video_tag

# storing the frames from training videos

# for i in tqdm(range(train.shape[0])):
#     count = 0
#     videoFile = train['video_name'][i]
#     cap = cv2.VideoCapture(sys.path[0] + "/media/out/train/" + videoFile)
#     success,image = cap.read()
#     while success:
#         cv2.imwrite(sys.path[0] + "/outputs/train_1/" + videoFile[:-4] +"_frame%d.jpg" % count, image)
#         success,image = cap.read()
#         count += 1
#     cap.release()

images = glob(sys.path[0] + "/outputs/train_1/*.jpg")
train_image = []
train_class = []

for i in tqdm(range(len(images))):
    # creating the image name
    train_image.append(images[i].split(sys.path[0] + "/outputs/train_1/")[1])
    # creating the class of image
    train_class.append(images[i].split(sys.path[0] + "/outputs/train_1/")[1].split('_')[0])

# storing the images and their class in a dataframe
train_data = pd.DataFrame()
train_data['image'] = train_image
train_data['class'] = train_class

train_data.to_csv(sys.path[0] + "/outputs/train_csv/train_new.csv",header=True, index=False)
