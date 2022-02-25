import os
import glob
import tensorflow.keras as keras
from tensorflow.keras.layers import TimeDistributed, GRU, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_video import VideoFrameGenerator
import tensorflow as tf

# use sub directories names as classes
classes = [i.split(os.path.sep)[1] for i in glob.glob('videos/*')]
classes.sort()

# some global params
SIZE = (224, 224)
CHANNELS = 3
NBFRAME = 20
BS = 10

# pattern to get videos and classes
glob_pattern='videos/{classname}/*.mp4'

# Create video frame generator
train = VideoFrameGenerator(
    classes=classes,
    glob_pattern=glob_pattern,
    nb_frames=NBFRAME,
    split_val=.2,
    shuffle=True,
    batch_size=BS,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    use_frame_cache=True)

valid = train.get_validation_generator()

# import keras_video.utils
# keras_video.utils.show_sample(train)

def build_mobilenet(shape=(224, 224, 3), nbout=3):
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

def action_model(shape=(20, 224, 224, 3), nbout=3):
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

INSHAPE=(NBFRAME,) + SIZE + (CHANNELS,) # (5, 224, 224, 3)

model = action_model(INSHAPE, len(classes))

# print(model.layers)

optimizer = keras.optimizers.SGD()

model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc']
)

EPOCHS=50

# create a "chkp" directory before to run that
# because ModelCheckpoint will write models inside

callbacks = [
    keras.callbacks.ReduceLROnPlateau(verbose=1),
    keras.callbacks.ModelCheckpoint(
        'chkp/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        verbose=1),
]

mcp_save = ModelCheckpoint(
    'weightSeq.hdf5', save_best_only=True, monitor='val_loss', mode='min')

model.fit(
    train,
    validation_data=valid,
    verbose=1,
    epochs=EPOCHS,
    callbacks=mcp_save
)
