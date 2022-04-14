from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import imutils
import cv2
import os
import mediapipe as mp
import sys
import numpy as np
import re
from shapely.geometry import Point, box
import threading
import pandas as pd

from tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from keras.callbacks import ModelCheckpoint

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper


class PhotoBoothApp:
    def __init__(self, vs, outputPath):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.SAMPLES = 25
        self.vs = vs
        self.outputPath = outputPath
        self.frame = None
        self.thread = None
        self.stopEvent = None
        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None

        f = sys.path[0] + "/keys.csv"
        self.keys = pd.read_csv(sys.path[0] + "/keys.csv")
        self.keys = self.keys.columns.tolist()

        self.var1 = tki.IntVar(value=1)
        self.var2 = tki.IntVar(value=1)
        c1 = tki.Checkbutton(self.root, text='Show Hand Meshes',
                             variable=self.var1, onvalue=1, offvalue=0)
        c1.grid(row=0, column=0, pady=2)
        c2 = tki.Checkbutton(self.root, text='Show Video',
                             variable=self.var2, onvalue=1, offvalue=0)
        c2.grid(row=0, column=1, pady=2)
        self.text1 = tki.StringVar()
        self.text2 = tki.StringVar()
        self.text3 = tki.StringVar()
        self.text4 = tki.StringVar()
        self.text5 = tki.StringVar()
        self.text6 = tki.StringVar()
        self.textpred = tki.StringVar()
        l1 = tki.Label(self.root, textvariable=self.text1, fg='#00f')
        l2 = tki.Label(self.root, textvariable=self.text2, fg='#055')
        l3 = tki.Label(self.root, textvariable=self.text3, fg='#404')
        l4 = tki.Label(self.root, textvariable=self.text4, fg='#00f')
        l5 = tki.Label(self.root, textvariable=self.text5, fg='#055')
        l6 = tki.Label(self.root, textvariable=self.text6, fg='#404')
        lpred = tki.Label(self.root, textvariable=self.textpred, fg='#000')
        
        l1.grid(row=2, column=0, pady=2)
        l2.grid(row=3, column=0, pady=2)
        l3.grid(row=4, column=0, pady=2)
        l4.grid(row=2, column=1, pady=2)
        l5.grid(row=3, column=1, pady=2)
        l6.grid(row=4, column=1, pady=2)
        lpred.grid(row=5, column=0, columnspan=2, pady=2)


        self.coord_line = [["", "", ""], ["", "", ""], [
            "", "", ""], ["", "", ""], ["", "", ""], ["", "", ""]]
        self.coord_float = []
        self.coord_pred = []
        self.pattern = r"[xyz]: "
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()
        # set a callback to handle when the window is closed
        self.root.wm_title("Demonstration Version")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
        self.action = run_once(self.printNums)
        self.recording = False

        self.model = Sequential()
        self.model.add(TCN(50, activation='relu',
                  input_shape=(self.SAMPLES, 18)))
        self.model.add(Dense(1))
        self.model.add(Dense(5, activation='softmax'))
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['acc'])
        # loading the trained weights
        self.model.load_weights(sys.path[0] + "/25_SAMPLES_ACC_76.0%.hdf5")

        # compiling the model
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['acc'])

    def videoLoop(self):
        # DISCLAIMER:
        # I'm not a GUI developer, nor do I even pretend to be. This
        # try/except statement is a pretty ugly hack to get around
        # a RunTime error that Tkinter throws due to threading
        with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            try:
                # keep looping over frames until we are instructed to stop
                while not self.stopEvent.is_set():
                    # grab the frame from the video stream and resize it to
                    # have a maximum width of 300 pixels
                    self.frame = self.vs.read()
                    self.image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

                    self.image = cv2.rotate(
                        self.image, cv2.cv2.ROTATE_90_CLOCKWISE)
                    # self.image = cv2.flip(self.image, 1)

                    # OpenCV represents images in BGR order; however PIL
                    # represents images in RGB order, so we need to swap
                    # the channels, then convert to PIL and ImageTk format

                    self.image.flags.writeable = False
                    results = hands.process(self.image)
                    self.image.flags.writeable = True
                    if (self.var2.get() == 0):
                        height, width, channels = self.image.shape
                        color = (0, 0, 0)
                        self.image = cv2.rectangle(
                            self.image, (0, 0), (width, height), color, 1000)

                    if(self.recording == False):
                        box_colour = (255, 0, 0)
                    else:
                        box_colour = (0, 255, 0)

                    height, width, channels = self.image.shape
                    rect_x = int(width * 0.2)
                    rect_y = int(height - rect_x)
                    self.image = cv2.rectangle(
                        self.image, (2, rect_y), (rect_x, height - 2), box_colour, 2)

                    if results.multi_hand_landmarks:
                        hand_number = 1
                        for hand_landmarks in results.multi_hand_landmarks:
                            if(hand_number == 1):
                                self.coord_line[0] = (
                                    str(hand_landmarks.landmark[0]).strip().split("\n"))
                                self.coord_line[1] = (
                                    str(hand_landmarks.landmark[4]).strip().split("\n"))
                                self.coord_line[2] = (
                                    str(hand_landmarks.landmark[8]).strip().split("\n"))
                            elif(hand_number == 2):
                                self.coord_line[3] = (
                                    str(hand_landmarks.landmark[0]).strip().split("\n"))
                                self.coord_line[4] = (
                                    str(hand_landmarks.landmark[4]).strip().split("\n"))
                                self.coord_line[5] = (
                                    str(hand_landmarks.landmark[8]).strip().split("\n"))

                            hand_number += 1

                            if (self.var1.get() == 1):
                                mp_drawing.draw_landmarks(
                                    self.image,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style())

                        tmp = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [
                            0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                        if("" not in self.coord_line[0]):
                            for idx1, x in enumerate(self.coord_line):
                                # print(coord_line[idx1])
                                for idx2, y in enumerate(x):
                                    self.coord_line[idx1][idx2] = re.sub(
                                        self.pattern, '', y)
                                    if(y != ""):
                                        tmp[idx1][idx2] = round(
                                            float(self.coord_line[idx1][idx2]), 5)
                                    else:
                                        tmp[idx1][idx2] = 0.0
                        self.coord_float = (tmp)
                        if(self.recording == True):
                            self.coord_pred.append(tmp)

                        self.text1.set("x:" + str(self.coord_float[0][0]) + " y:" + str(
                            self.coord_float[0][1]) + " z:" + str(self.coord_float[0][2]))
                        self.text2.set("x:" + str(self.coord_float[1][0]) + " y:" + str(
                            self.coord_float[1][1]) + " z:" + str(self.coord_float[1][2]))
                        self.text3.set("x:" + str(self.coord_float[2][0]) + " y:" + str(
                            self.coord_float[2][1]) + " z:" + str(self.coord_float[2][2]))
                        self.text4.set("x:" + str(self.coord_float[3][0]) + " y:" + str(
                            self.coord_float[3][1]) + " z:" + str(self.coord_float[3][2]))
                        self.text5.set("x:" + str(self.coord_float[4][0]) + " y:" + str(
                            self.coord_float[4][1]) + " z:" + str(self.coord_float[4][2]))
                        self.text6.set("x:" + str(self.coord_float[5][0]) + " y:" + str(
                            self.coord_float[5][1]) + " z:" + str(self.coord_float[5][2]))

                        poly_path = box(0, rect_y, rect_x, height - 2)

                        # int_coords = lambda x: np.array(x).round().astype(np.int32)
                        # exterior = [int_coords(poly_path.exterior.coords)]
                        # self.image = cv2.fillPoly(self.image, exterior, color=(0, 255, 0))

                        p1 = Point(
                            int(self.coord_float[2][0] * width), int(self.coord_float[2][1] * height))

                        if(p1.within(poly_path)):
                            self.action()

                    if(self.action.has_run == True):
                        if(self.timerCount >= 0):
                            # setup text
                            font = cv2.FONT_HERSHEY_PLAIN
                            text = str(self.timerCount)
                            # get boundary of this text
                            textsize = cv2.getTextSize(text, font, 20, 2)[0]
                            # get coords based on boundary
                            textX = int((width - textsize[0]) / 2)
                            textY = int((height + textsize[1]) / 2)
                            # add text centered on image
                            self.image = cv2.putText(
                                self.image, text, (textX, textY), font, 20, (255, 255, 255), 2)

                    self.image = Image.fromarray(self.image)
                    self.image = ImageTk.PhotoImage(self.image)

                    # if the panel is not None, we need to initialize it
                    if self.panel is None:
                        self.panel = tki.Label(image=self.image)
                        self.panel.image = self.image
                        self.panel.grid(row=1, column=0, columnspan=2, pady=2)

                    # otherwise, simply update the panel
                    else:
                        self.panel.configure(image=self.image)
                        self.panel.image = self.image
            except(RuntimeError):
                print("[INFO] caught a RuntimeError")

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()

    def printNums(self):
        self.timerCount = 3

        def predict():
            diff_line = []
            for idx1, x in enumerate(self.coord_pred):
                tmp = []
                for idx2, y in enumerate(x):
                    for idx3, z in enumerate(y):
                        if(idx1 + 1 < len(self.coord_pred) and idx2 + 1 < 12 and idx3 < 6):
                            tmp.append(abs(
                                self.coord_pred[idx1][idx2][idx3]) - abs(self.coord_pred[idx1 + 1][idx2][idx3]))
                diff_line.append(tmp)

            diff_line = list(filter(None, diff_line))
            diff_line = np.asarray(diff_line)

            idx = np.round(np.linspace(
                0, diff_line.shape[0] - 1, self.SAMPLES)).astype(int)
            diff_line = diff_line[idx]
            # print(np.shape(diff_line))

            pred = self.model.predict(diff_line.reshape((1, self.SAMPLES, 18)))
            pred = np.argmax(pred, axis=1)
            self.textpred.set(self.keys[pred[0]])

        def rec():
            timer = threading.Timer(1.0, rec)
            if(self.timerCount > 0):
                timer.start()
                self.timerCount -= 1
            else:
                self.recording = False
                predict()
                return

        def printit():
            timer = threading.Timer(1.0, printit)
            if(self.timerCount > 0):
                timer.start()
                self.timerCount -= 1
            else:
                self.action.has_run = False
                self.recording = True
                self.timerCount = 3
                rec()
                return

        printit()


if __name__ == "__main__":
    from imutils.video import VideoStream
    import argparse
    import time
    # initialize the video stream and allow the camera sensor to warmup
    print("[INFO] warming up camera...")
    vs = VideoStream(src=0, usePiCamera=False > 0).start()
    time.sleep(2.0)
    # start the app
    pba = PhotoBoothApp(vs, sys.path[0])
    pba.root.mainloop()
