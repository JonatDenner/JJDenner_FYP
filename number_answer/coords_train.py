import numpy as np
import os
import sys
import re

from collections import OrderedDict

import random

from tcn import TCN, tcn_full_summary

from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential

from sklearn.metrics import accuracy_score

train_path = sys.path[0] + "/csv_out/train/"
test_path = sys.path[0] + "/csv_out/test/"
train_files = os.listdir(train_path)
test_files = os.listdir(test_path)
#random.shuffle(train_files)
#random.shuffle(test_files)
train_csv = []
test_csv = []


SAMPLE_SIZE = 25

X = []
classes = []

test_input = []
test_actual = []

for x in test_files:
    if x.endswith(".csv"):
        test_csv.append(x)

for x in train_files:
    if x.endswith(".csv"):
        train_csv.append(x)

for f in train_csv:
    filename = re.sub(r"_.*\.csv", "", f)
    filename = re.sub(r"(\w)([A-Z])", r"\1 \2", filename)
    temp = np.loadtxt(train_path + f, delimiter=',')

    X_min = temp.min(axis=(0, 1), keepdims=True)
    X_max = temp.max(axis=(0, 1), keepdims=True)
    temp = (temp - X_min)/(X_max - X_min)

    X.append(temp)
    classes.append(filename)

for f in test_csv:
    filename = re.sub(r"_.*\.csv", "", f)
    filename = re.sub(r"(\w)([A-Z])", r"\1 \2", filename)
    temp = np.loadtxt(test_path + f, delimiter=',')

    X_min = temp.min(axis=(0, 1), keepdims=True)
    X_max = temp.max(axis=(0, 1), keepdims=True)
    temp = (temp - X_min)/(X_max - X_min)

    test_input.append(temp)
    test_actual.append(filename)


y = [{v: k for k, v in enumerate(
   OrderedDict.fromkeys(classes))}
      [n] for n in classes]

test_actual = [{v: k for k, v in enumerate(
   OrderedDict.fromkeys(test_actual))}
      [n] for n in test_actual]

#print(list(zip(classes, y)))

classes = list(dict.fromkeys(classes))
keys = list(dict.fromkeys(y))
keys = dict(zip(keys, classes))
#print(keys)

X = np.asarray(X)

y = np.asarray(y)

#print(X)

# model = Sequential()
# model.add(LSTM(50, activation='relu', input_shape=(SAMPLE_SIZE, 18)))
# model.add(Dense(1))
# model.add(Dense(len(keys), activation='linear'))
# model.compile(optimizer='adam', loss='mse')

model = Sequential()
model.add(TCN(50, nb_stacks=1, activation='relu', input_shape=(SAMPLE_SIZE, 18)))
model.add(Dense(1))
model.add(Dense(len(keys)))
model.compile(optimizer='adam', loss='mse')

#print(model.summary())

model.fit(X, y, epochs=1000, validation_split=0.2, verbose=2)

test_input = np.asarray(test_input)

prediction = []

print(keys)

for x in test_input:
    pred = model.predict(x.reshape((1, SAMPLE_SIZE, 18)))
    classes_pred=np.argmax(pred,axis=1)
    print(pred)
    prediction.append(classes_pred[0])


score = accuracy_score(prediction, test_actual)*100
print("\033[1;36m----------------------------------\nAccuracy:")
print(score)

predictions = list(zip(prediction, test_actual))
for x in predictions:
    p = x[0]
    a = x[1]

    if p == a:
        print("\033[1;32mPrediction matched result: " + keys[p])
    else:
        print("\033[1;31mPrediction: " + keys[p] + "\t\tActual: " + keys[a])
