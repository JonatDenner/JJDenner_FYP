import numpy as np
import os
import sys
import re
import pandas as pd
from scipy import stats as s

from collections import OrderedDict

import random

from tcn import TCN, tcn_full_summary

from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import accuracy_score

train_path = sys.path[0] + "/csv_out/train/"
val_path = sys.path[0] + "/csv_out/validate/"
test_path = sys.path[0] + "/csv_out/test/"
train_files = os.listdir(train_path)
test_files = os.listdir(test_path)
val_files = os.listdir(val_path)
train_csv = []
test_csv = []
val_csv = []


SAMPLE_SIZE = 15

X = []
X_val = []
classes_val = []
classes = []

test_input = []
test_actual = []

for x in test_files:
    if x.endswith(".csv"):
        test_csv.append(x)

for x in val_files:
    if x.endswith(".csv"):
        val_csv.append(x)

for x in train_files:
    if x.endswith(".csv"):
        train_csv.append(x)

for f in train_csv:
    filename = re.sub(r"_.*\.csv", "", f)
    filename = re.sub(r"(\w)([A-Z])", r"\1 \2", filename)
    temp = np.loadtxt(train_path + f, delimiter=',')

    # X_min = temp.min(axis=(0, 1), keepdims=True)
    # X_max = temp.max(axis=(0, 1), keepdims=True)
    # temp = (temp - X_min)/(X_max - X_min)

    X.append(temp)
    classes.append(filename)

for f in val_csv:
    filename = re.sub(r"_.*\.csv", "", f)
    filename = re.sub(r"(\w)([A-Z])", r"\1 \2", filename)
    temp = np.loadtxt(val_path + f, delimiter=',')

    # X_min = temp.min(axis=(0, 1), keepdims=True)
    # X_max = temp.max(axis=(0, 1), keepdims=True)
    # temp = (temp - X_min)/(X_max - X_min)

    X_val.append(temp)
    classes_val.append(filename)

for f in test_csv:
    filename = re.sub(r"_.*\.csv", "", f)
    filename = re.sub(r"(\w)([A-Z])", r"\1 \2", filename)
    temp = np.loadtxt(test_path + f, delimiter=',')

    # X_min = temp.min(axis=(0, 1), keepdims=True)
    # X_max = temp.max(axis=(0, 1), keepdims=True)
    # temp = (temp - X_min)/(X_max - X_min)

    test_input.append(temp)
    test_actual.append(filename)


# y = [{v: k for k, v in enumerate(
#    OrderedDict.fromkeys(classes))}
#       [n] for n in classes]

# test_actual = [{v: k for k, v in enumerate(
#    OrderedDict.fromkeys(test_actual))}
#       [n] for n in test_actual]

#print(list(zip(classes, y)))

y = pd.get_dummies(classes) # sklearn one hot encode
y_val = pd.get_dummies(classes_val)

keys = list(dict.fromkeys(y))
with open('number_answer/keys.csv', 'w') as f:
    output = ""
    for x in keys:
        output = output + str(x) + ","
    f.write(output.rstrip(','))
f.close()
#print(keys)

X = np.asarray(X)
X_val = np.asarray(X_val)



#print(X)

model = Sequential()
model.add(TCN(50, activation='relu', input_shape=(SAMPLE_SIZE, 18)))
model.add(Dense(1))
model.add(Dense(len(keys), activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#print(model.summary())
checkpoint_name = sys.path[0] + "/tmp.hdf5"
mcp_save = ModelCheckpoint(
    checkpoint_name, save_best_only=True, monitor='val_loss', mode='min')

model.fit(X, y, epochs=1000, validation_data=(X_val, y_val), verbose=2, callbacks=[mcp_save])

test_input = np.asarray(test_input)

prediction = []

#print(keys)

for x in test_input:
    pred = model.predict(x.reshape((1, SAMPLE_SIZE, 18)))
    classes_x=np.argmax(pred,axis=1)
    prediction.append(y.columns.values[s.mode(classes_x)[0][0]])

print(prediction)
print(test_actual)
score = accuracy_score(prediction, test_actual)*100
model_acc = sys.path[0] + "/" + str(SAMPLE_SIZE) + "_SAMPLES_ACC_" + str(round(score)) + "%.hdf5"
os.rename(checkpoint_name, model_acc)
print("\033[1;36m----------------------------------\nAccuracy:")
print(score)

predictions = list(zip(prediction, test_actual))
for x in predictions:
    p = x[0]
    a = x[1]

    if p == a:
        print("\033[1;32mPrediction matched result: " + p)
    else:
        print("\033[1;31mPrediction: " + p + "\t\tActual: " + a)
