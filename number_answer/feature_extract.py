import cv2
import mediapipe as mp
import re
import numpy as np
import sys
import os


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

main_folder_path = sys.path[0] + "/media/1/train/"
output_folder_path = sys.path[0] + "/csv_out/train/"

SAMPLES = 15

def extract(f, count):
    print(str(count), " extracting ", f)
    cap = cv2.VideoCapture(main_folder_path + f)

    filename = re.sub(r".mp4", "", f)

    pattern = r"[xyz]: "

    coord_line = [["", "", ""],["", "", ""],["", "", ""],["", "", ""],["", "", ""],["", "", ""]]
    csv_line = []
    diff_line = []

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



        if results.multi_hand_landmarks:
          hand_number = 1
          for hand_landmarks in results.multi_hand_landmarks:

            # 0 hand, 4 thumb, 8 index
            if(hand_number == 1):
                coord_line[0] = (str(hand_landmarks.landmark[0]).strip().split("\n"))
                coord_line[1] = (str(hand_landmarks.landmark[4]).strip().split("\n"))
                coord_line[2] = (str(hand_landmarks.landmark[8]).strip().split("\n"))
            elif(hand_number == 2):
                coord_line[3] = (str(hand_landmarks.landmark[0]).strip().split("\n"))
                coord_line[4] = (str(hand_landmarks.landmark[4]).strip().split("\n"))
                coord_line[5] = (str(hand_landmarks.landmark[8]).strip().split("\n"))

            hand_number += 1

            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

          tmp = [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]
          if("" not in coord_line[0]):
              for idx1, x in enumerate(coord_line):
                  #print(coord_line[idx1])
                  for idx2, y in enumerate(x):
                      coord_line[idx1][idx2] = re.sub(pattern, '', y)
                      if(y != ""):
                          tmp[idx1][idx2] = float(coord_line[idx1][idx2])
                      else:
                          tmp[idx1][idx2] = 0.0
          csv_line.append(tmp)

    cap.release()

    for idx1, x in enumerate(csv_line):
        tmp = []
        for idx2, y in enumerate(x):
            for idx3, z in enumerate(y):
                if(idx1+1 < len(csv_line) and idx2 + 1 < 12 and idx3 < 6):
                    tmp.append(abs(csv_line[idx1][idx2][idx3]) - abs(csv_line[idx1+1][idx2][idx3]))
        diff_line.append(tmp)

    diff_line = list(filter(None, diff_line))
    diff_line = np.asarray(diff_line)

    idx = np.round(np.linspace(0, diff_line.shape[0] - 1, SAMPLES)).astype(int)
    diff_line = diff_line[idx]

    np.savetxt(output_folder_path + filename + ".csv", diff_line, delimiter=",")

if __name__ == "__main__":
    total_files = os.listdir(main_folder_path)
    total_files_m = []

    for x in total_files:
        if x.endswith(".mp4"):
            #name = re.sub(r"_.*\.mp4", "", x)
            total_files_m.append(x)

    for count, x in enumerate(total_files_m):
        extract(x, count)
