import mediapipe as mp
import cv2
import os
import sys
import multiprocessing as multi

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def main_process(filename):
    frameNumber = 0
    if filename.endswith(".mp4"):
        f = open('outputs/' + filename[:-4] + '.csv', 'w')
        f.write("frame,hand_number,wrist_x,wrist_y,wrist_z,thumb_cmc_x,thumb_cmc_y,thumb_cmc_z,thumb_mcp_x,thumb_mcp_y,thumb_mcp_z,thumb_ip_x,thumb_ip_y,thumb_ip_z,thumb_tip_x,thumb_tip_y,thumb_tip_z,index_finger_mcp_x,index_finger_mcp_y,index_finger_mcp_z,index_finger_pip_x,index_finger_pip_y,index_finger_pip_z,index_finger_dip_x,index_finger_dip_y,index_finger_dip_z,index_finger_tip_x,index_finger_tip_y,index_finger_tip_z,middle_finger_mcp_x,middle_finger_mcp_y,middle_finger_mcp_z,middle_finger_pip_x,middle_finger_pip_y,middle_finger_pip_z,middle_finger_dip_x,middle_finger_dip_y,middle_finger_dip_z,middle_finger_tip_x,middle_finger_tip_y,middle_finger_tip_z,ring_finger_mcp_x,ring_finger_mcp_y,ring_finger_mcp_z,ring_finger_pip_x,ring_finger_pip_y,ring_finger_pip_z,ring_finger_dip_x,ring_finger_dip_y,ring_finger_dip_z,ring_finger_tip_x,ring_finger_tip_y,ring_finger_tip_z,pinky_mcp_x,pinky_mcp_y,pinky_mcp_z,pinky_pip_x,pinky_pip_y,pinky_pip_z,pinky_dip_x,pinky_dip_y,pinky_dip_z,pinky_tip_x,pinky_tip_y,pinky_tip_z\n")
        cap = cv2.VideoCapture("media/out/" + filename)
        print(filename)
        sys.stdout.flush()
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter("media/curr/"+ filename[:-4] +".avi", fourcc, 5, (1600, 1280))

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
                    print("Finished.")
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
                        print(rect)
                image_height, image_width, _ = image.shape
                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    hand_number = 0
                    for hand_landmarks in results.multi_hand_landmarks:
                        f.write(str(frameNumber))
                        f.write(str(hand_number) + ",")
                        for x in hand_landmarks.landmark:
                            f.write(str(x.x) + "," + str(x.y) +
                                    "," + str(x.z) + ",")
                        f.write("\n")
                        hand_number += 1
                frameNumber += 1
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        cap.release()
        f.close()
        os.remove("media/curr/"+ filename[:-4] +".avi")

if __name__ == "__main__":
    total_files = os.listdir(sys.path[0] + "/media/out")
    thread_count = 5
    iterations = len(total_files)//thread_count
    pool = multi.Pool(multi.cpu_count())
    # example
    # 250/5 = 50
    # -> 50 runs required
    # -> overall range: 50 >> iterations
    # -> range inside range: 5 >> thread_count
    # thread_count has to be dividable by the total number of videos
    for x in range(iterations):
        print("Iteration: "+str(x))
        video_number = x*thread_count
        pool.map(main_process, [total_files[video_number+y] for y in range(thread_count)])
