import os
import glob
import tensorflow.keras as keras
from tensorflow.keras.layers import TimeDistributed, GRU, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_video import VideoFrameGenerator
import tensorflow as tf
import numpy as np

classes = [i.split(os.path.sep)[1] for i in glob.glob('videos/*')]
classes.sort()

SIZE = (224, 224)
CHANNELS = 3
NBFRAME = 25
BS = 15

glob_pattern='test/{classname}/*.mp4'

test = VideoFrameGenerator(
    classes=classes,
    glob_pattern=glob_pattern,
    nb_frames=NBFRAME,
    shuffle=False,
    batch_size=BS,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    use_frame_cache=True)

# import keras_video.utils
# keras_video.utils.show_sample(test)

def build_mobilenet(shape=(224, 224, 3), nbout=5):
    model = keras.applications.mobilenet.MobileNet(
        include_top=False,
        input_shape=shape,
        weights='imagenet')

    # Keep 9 layers to train﻿﻿
    trainable = 9
    for layer in model.layers[:-trainable]:
        layer.trainable = False
    for layer in model.layers[-trainable:]:
        layer.trainable = True

    output = keras.layers.GlobalMaxPool2D()
    return keras.Sequential([model, output])

def action_model(shape=(25, 224, 224, 3), nbout=5):
    # Create our convnet with (112, 112, 3) input shape
    convnet = build_mobilenet(shape[1:])

    # then create our final model
    model = keras.Sequential()

        # add the convnet with (5, 112, 112, 3) shape
    model.add(TimeDistributed(convnet, input_shape=shape))    # here, you can also use GRU or LSTM
    model.add(GRU(64))    # and finally, we make a decision network
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nbout, activation='softmax'))
    return model

INSHAPE=(NBFRAME,) + SIZE + (CHANNELS,)

model = action_model(INSHAPE, len(classes))

model.load_weights("weightSeq.hdf5")

optimizer = "Adam"

model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc']
)

result = model.predict(test, verbose=1)
#print(test.classes)

# print(classes)
for x in result:
    print(classes[np.argmax(x)])
