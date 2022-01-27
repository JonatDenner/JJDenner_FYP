import mediapipe as mp
import cv2
import os
import sys
import multiprocessing as multi
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def main_process(filename):
    frameNumber = 0
    coordinates = []
    if filename.endswith(".mp4"):
        # f = open('outputs/' + filename[:-4] + '.csv', 'w')
        # f.write("frame,hand_number,wrist_x,wrist_y,wrist_z,thumb_cmc_x,thumb_cmc_y,thumb_cmc_z,thumb_mcp_x,thumb_mcp_y,thumb_mcp_z,thumb_ip_x,thumb_ip_y,thumb_ip_z,thumb_tip_x,thumb_tip_y,thumb_tip_z,index_finger_mcp_x,index_finger_mcp_y,index_finger_mcp_z,index_finger_pip_x,index_finger_pip_y,index_finger_pip_z,index_finger_dip_x,index_finger_dip_y,index_finger_dip_z,index_finger_tip_x,index_finger_tip_y,index_finger_tip_z,middle_finger_mcp_x,middle_finger_mcp_y,middle_finger_mcp_z,middle_finger_pip_x,middle_finger_pip_y,middle_finger_pip_z,middle_finger_dip_x,middle_finger_dip_y,middle_finger_dip_z,middle_finger_tip_x,middle_finger_tip_y,middle_finger_tip_z,ring_finger_mcp_x,ring_finger_mcp_y,ring_finger_mcp_z,ring_finger_pip_x,ring_finger_pip_y,ring_finger_pip_z,ring_finger_dip_x,ring_finger_dip_y,ring_finger_dip_z,ring_finger_tip_x,ring_finger_tip_y,ring_finger_tip_z,pinky_mcp_x,pinky_mcp_y,pinky_mcp_z,pinky_pip_x,pinky_pip_y,pinky_pip_z,pinky_dip_x,pinky_dip_y,pinky_dip_z,pinky_tip_x,pinky_tip_y,pinky_tip_z\n")
        cap = cv2.VideoCapture("media/out/aug/" + filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        #print(filename)
        sys.stdout.flush()
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter("media/curr/"+ filename[:-4] +".avi", fourcc, fps, (1600, 1280))

        while True:
            ret, frame = cap.read()
            if ret == True:
                b = cv2.resize(frame, (1600, 1280), fx=0, fy=0,
                               interpolation=cv2.INTER_CUBIC)
                out.write(b)
            else:
                break

        cap.release()
        out.release()

        cap = cv2.VideoCapture("media/curr/"+ filename[:-4] +".avi")

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


        pixel_coords = [(coords[0]*1600).astype(int), (coords[1]*1280).astype(int), (coords[2]*1600).astype(int), (coords[3]*1280).astype(int)]
        height = pixel_coords[3] - pixel_coords[1]
        width = pixel_coords[2] - pixel_coords[0]
        #print(height)
        #print(width)

        current_file = sys.path[0] + "/media/curr/" + filename[:-4] + ".avi"
        current_file = current_file.replace(" ", "\ ")
        output_video = (sys.path[0] + "/media/out/" + filename).replace(" ", "\ ")

        #print("ffmpeg -i "+ current_file + " -filter:v 'crop=" + str(width) + ":" + str(height) + ":" + str(pixel_coords[0]) + ":" + str(pixel_coords[1]) + "' " + current_file[:-4] + ".mp4")
        os.system("ffmpeg -y -i "+ current_file + " -filter:v 'crop=" + str(width) + ":" + str(height) + ":" + str(pixel_coords[0]) + ":" + str(pixel_coords[1]) + "' " + output_video)

        #print(pixel_coords)


        print("Finished " + filename)
        os.remove("media/curr/"+ filename[:-4] +".avi")

if __name__ == "__main__":
    total_files = os.listdir(sys.path[0] + "/media/out/aug/")
    total_files_m = []

    for x in total_files:
        if x.endswith(".mp4"):
            total_files_m.append(x)

    thread_count = 5
    print("total videos to crop: " + str(len(total_files_m)))
    iterations = len(total_files_m)//thread_count
    pool = multi.Pool(multi.cpu_count())

    # example
    # 250/5 = 50
    # -> 50 runs required
    # -> overall range: 50 >> iterations
    # -> range inside range: 5 >> thread_count
    # thread_count has to be dividable by the total number of videos

    for x in range(iterations):
        #print("Iteration: "+str(x))
        video_number = x*thread_count
        pool.map(main_process, [total_files_m[video_number+y] for y in range(thread_count)])
