import mediapipe as mp
import cv2
import os
import sys
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def resize(filename):
    coordinates = []
    if filename.endswith(".mp4"):

        cap = cv2.VideoCapture(sys.path[0] + "/media/1/train/" + filename)
        video_width  = cap.get(3)
        video_height = cap.get(4)

        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    sys.stdout.flush()
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                if results.hand_rects:
                    for rect in results.hand_rects:
                        cd = []
                        cd.append(rect.x_center)
                        cd.append(rect.y_center)
                        cd.append(rect.height)
                        cd.append(rect.width)
                        coordinates.append(cd)

        cap.release()

        coords = [0, 0, 0, 0]
        a = np.array(coordinates)
        min_index = np.where(a == np.amin(a, axis=0))
        max_index = np.where(a == np.amax(a, axis=0))

        min_coord_list = list(zip(min_index[0], min_index[1]))
        max_coord_list = list(zip(max_index[0], max_index[1]))

        for x in min_coord_list:
            if x[1] == 0:
                coords[0] = a[x[0]][0] - a[x[0]][3]/2
            elif x[1] == 1:
                coords[1] = a[x[0]][1] - a[x[0]][2]/2

        for x in max_coord_list:
            if x[1] == 0:
                coords[2] = a[x[0]][0] + a[x[0]][3]/2
            elif x[1] == 1:
                coords[3] = a[x[0]][1]


        pixel_coords = [abs((coords[0]*video_width).astype(int)), abs((coords[1]*video_height).astype(int)), (coords[2]*video_width).astype(int), (coords[3]*video_height).astype(int)]
        height = pixel_coords[3] - pixel_coords[1]
        width = pixel_coords[2] - pixel_coords[0]
        #print(height)
        #print(width)

        current_file = sys.path[0] + "/media/1/train/" + filename
        current_file = current_file.replace(" ", "\ ")
        output_video = (sys.path[0] + "/media/2/train/" + filename).replace(" ", "\ ")

        #print("ffmpeg -i "+ current_file + " -filter:v 'crop=" + str(width) + ":" + str(height) + ":" + str(abs(pixel_coords[0])) + ":" + str(abs(pixel_coords[1])) + "' " + current_file[:-4] + ".mp4")
        os.system("ffmpeg -y -i "+ current_file + " -filter:v 'crop=" + str(width) + ":" + str(height) + ":" + str(abs(pixel_coords[0])) + ":" + str(abs(pixel_coords[1])) + "' " + output_video)

        #print(pixel_coords)


        print("Finished " + filename + "\n")

if __name__ == "__main__":
    total_files = os.listdir(sys.path[0] + "/media/1/train/")
    total_files_m = []

    for x in total_files:
        if x.endswith(".mp4"):
            total_files_m.append(x)

    for x in total_files_m:
        resize(x)
