import math
import random
import time
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import pygame
from datetime import datetime

score_board = pd.read_csv('./scoreboard.csv')

cv2.namedWindow('DOODEOJI', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('DOODEOJI', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

def main_home():
    pass

def random_pos():
    x = np.random.randint(150, 1130)
    y = np.random.randint(50, 670)
    return (x,y)

pygame.init()
whack_sound = pygame.mixer.Sound('./sound/boing.wav')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



# 투명한 영역이 있는 이미지 영상에 오버레이하는 함수
def overlay(image, pos, w, h, overlay_image): # 대상 이미지 (3채널), x, y 좌표, width, height, 덮어씌울 이미지 (4채널)
    alpha = overlay_image[:, :, 3] # BGRA
    mask_image = alpha / 255 # 0 ~ 255 -> 255 로 나누면 0 ~ 1 사이의 값 (1: 불투명, 0: 완전)
    for c in range(0, 3): # channel BGR
        image[pos[1]-h:pos[1]+h, pos[0]-w:pos[0]+w, c] = (overlay_image[:, :, c] * mask_image) + (image[pos[1]-h:pos[1]+h, pos[0]-w:pos[0]+w, c] * (1 - mask_image))




# 두 포인트 사이 거리 구하는 함수
def get_distance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

class mole:
    mole_image = cv2.imread('./image/mole_tr100.png', cv2.IMREAD_UNCHANGED)
    moleh, molew, _ = mole_image.shape
    def __init__(self, pos) -> None:
        self.x=pos[0]
        self.y=pos[1]
                

game_start_event = False
game_over_event = False
game_pause_event = False
score_recorded  = False




time_given=30.9
time_remaining = 99



score = 0

moles = [mole(random_pos())]
moles.append(mole(random_pos()))





# 이미지 선언
mole_image = cv2.imread('./image/mole_tr100.png', cv2.IMREAD_UNCHANGED)
moleh, molew, _ = mole_image.shape

shine_image = cv2.imread('./image/shine.png', cv2.IMREAD_UNCHANGED)
shineh, shinew, _ = shine_image.shape

clap_image = cv2.imread('./image/clap.png', cv2.IMREAD_UNCHANGED)
claph, clapw, _ = clap_image.shape

body_image = cv2.imread('./image/body.png', cv2.IMREAD_UNCHANGED)
bodyh, bodyw, _ = body_image.shape


# For webcam input:

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.flip(frame, 1)

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape 
        
        present_time = time.time()
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            rightindex = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y, landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].z]
            leftindex = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].z]       

            righthand = [rightindex[0]*w, rightindex[1]*h]
            lefthand = [leftindex[0]*w, leftindex[1]*h]          

            rightz = rightindex[2]*100
            leftz =  leftindex[2]*100


            noseindex = [landmarks[0].x, landmarks[0].y]
            nose = [noseindex[0]*w, noseindex[1]*h]

            rightfootindex = [landmarks[31].x, landmarks[31].y]
            leftfootindex = [landmarks[32].x, landmarks[32].y]
            rightfoot = [rightfootindex[0]*w, rightfootindex[1]*h]
            leftfoot = [leftfootindex[0]*w, leftfootindex[1]*h]



            if game_start_event == False:

                cv2.putText(image, 'Clap to start a Game',
                            (w//2-300, h//2-85),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (51, 102, 153), 3, cv2.LINE_AA)
                cv2.putText(image, 'Please keep some distance or adjust your webcam',
                            (w//2-210, h//2+210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, 'to show your whole body in camera frame',
                            (w//2-180, h//2+230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                overlay(image, (w//2-230, h//2-185), 50, 50, clap_image)
                overlay(image, (w//2, h//2+20), 112, 210, body_image)
                # print('거리', get_distance(righthand, lefthand))

                if get_distance(righthand, lefthand) < 30 and abs(leftz-rightz) < 20 and ( w//2-150 < nose[0] < w//2+150 and 10 < nose[1] < h//2+20):
                    game_start_event = True
                    start_time = time.time()
                    




            if game_start_event == True and time_remaining > 0:
                
                time_remaining = int(time_given - (present_time - start_time))
                delete_idx = []
                for idx, i in enumerate(moles):
                    if (i.x-50 < righthand[0] < i.x+50 and i.y-50 < righthand[1] < i.y+50) or (i.x-50 < lefthand[0] < i.x+50 and i.y-50 < lefthand[1] < i.y+50) or (i.x-50 < rightfoot[0] < i.x+50 and i.y-50 < rightfoot[1] < i.y+50) or (i.x-50 < leftfoot[0] < i.x+50 and i.y-50 < leftfoot[1] < i.y+50):
                        overlay(image, (i.x, i.y), 50, 50, shine_image)
                        score += 1
                        whack_sound.play()
                        moles.append(mole(random_pos()))
                        delete_idx.append(idx)
                delete_idx.reverse()
                for i in delete_idx:
                    del moles[i]

                cv2.putText(image, 'Score:',
                            (w//2-250, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 204), 2, cv2.LINE_AA)      

                cv2.putText(image, str(score),
                           (w//2-130, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 204), 2, cv2.LINE_AA)         

                cv2.putText(image, 'Time left:',
                          (w//2+30, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 204), 2, cv2.LINE_AA)      

                cv2.putText(image, str(time_remaining),
                         (w//2+230, 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 204), 2, cv2.LINE_AA) 



                # image, x, y, w, h, overlay_image (좌측 최상단x,y가 50, 50임) 최하단 430 최우측 590
                for i in moles:
                    overlay(image,( i.x, i.y), 50, 50, mole_image)
            
            elif game_start_event == True and time_remaining <= 0:
                
                time_remaining = 0
                game_over_event = True





        except:
            cv2.putText(image, 'Please show your face and keep some distance from your webcam',
            (w//2-260, h//2+220),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            pass    

        # 게임종료시에만 실행
        if game_over_event == True:
            
            cv2.rectangle(image, (w//2-170, h//2-130), (w//2+170, h//2+40), (0,0,0), -1)

            cv2.putText(image, 'Game Over',
            (w//2-147, h//2-65),
            cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3, cv2.LINE_AA)
            
            cv2.putText(image, 'Your Score:',
            (w//2-120, h//2),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 

            cv2.putText(image, str(score),
            (w//2+80, h//2+3),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)   
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            if not score_recorded:
                score_board = pd.concat([score_board, pd.DataFrame({'name':['Unknown'], 'score' : [score], 'time':[formatted_time]})], ignore_index=True)
                score_board.to_csv('./scoreboard.csv',index=False)
                score_recorded = True

        # Render detections

        # mp_drawing.draw_landmarks(
        #     image,
        #     results.pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())  



        cv2.imshow('DOODEOJI', image)  # 화면크기 2배 키움
        command = cv2.waitKey(5) & 0xFF
        if command == 27:
            break
        elif command == ord('r') or command == ord('R'):
            game_start_event = False
            game_over_event = False
            game_pause_event = False
            score_recorded = False

            time_given=30.9
            time_remaining = 99

            score = 0
            continue
cap.release()