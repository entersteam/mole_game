import math
import time
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import pygame
from datetime import datetime

BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)
ORANGE = (0,165,255)

score_board = pd.read_csv('./scoreboard.csv')
ranking = score_board.sort_values(by='score', ascending=False)

cv2.namedWindow('DOODEOJI', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('DOODEOJI', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#cv2.moveWindow('DOODEOJI', 1920, 0)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)


height = 720

def random_pos():
    x = np.random.randint(178, 1102)
    y = np.random.randint(770 - height, 670)
    return (x,y)

def unit_vector(vector):
    unit = vector / np.linalg.norm(vector)
    return unit

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

#두더지 클래스
class mole:
    def __init__(self, pos, life=5.0, text=None) -> None:
        self.texture = np.random.randint(3)
        self.x=pos[0]
        self.y=pos[1]
        self.life = life
        self.spawn_time = time.time()
        self.text = text
    def check(self, target_list):
        if time.time() - self.spawn_time > self.life:
            target_list.pop(0)
            #moles.append(mole(random_pos()))

class elite_mole:    
    def __init__(self, pos) -> None:
        self.x=pos[0]
        self.y=pos[1]
        self.life = 3
        self.spawn_time = time.time()
        self.speed = 20
        self.vector = unit_vector(np.random.normal(0, 1, 2))
            
    def reflect(self, norm):
        self.vector += 2 * norm * np.dot(norm,-self.vector)
        
    def crash_detection(self):
        if self.x>1102:
            norm = np.array((-1,0))
            self.reflect(norm)
        if self.x<178:
            norm = np.array((1,0))
            self.reflect(norm)
        if self.y> 670:
            norm = np.array((0,1))
            self.reflect(norm)
        if self.y< 770-height:
            norm = np.array((0,-1))
            self.reflect(norm)
        
    def move(self):
        pos = self.vector*self.speed + np.array([self.x,self.y])
        self.x = pos[0]
        self.y = pos[1]
        
    def check(self, target_list):
        if time.time() - self.spawn_time > self.life:
            target_list.pop(0)
            #moles.append(mole(random_pos()))              

game_start_event = False
countdown = False
game_over_event = False
game_pause_event = False
score_recorded  = False

respawn_time = -np.Inf
respawning_time = 0.4

time_given=30.9
time_remaining = 99


score = 0.1

moles = [mole(random_pos())]
moles.append(mole(random_pos()))
elites = [elite_mole(random_pos())]
texts = []


# 이미지 선언
mole_image = cv2.imread('./image/mole_tr100.png', cv2.IMREAD_UNCHANGED)
moleh, molew, _ = mole_image.shape

shine_image = cv2.imread('./image/shine.png', cv2.IMREAD_UNCHANGED)
shineh, shinew, _ = shine_image.shape

clap_image = cv2.imread('./image/clap.png', cv2.IMREAD_UNCHANGED)
claph, clapw, _ = clap_image.shape

body_image = cv2.imread('./image/body.png', cv2.IMREAD_UNCHANGED)
bodyh, bodyw, _ = body_image.shape

lion_image = cv2.imread('./image/lion.png', cv2.IMREAD_UNCHANGED)
lion_image = cv2.resize(lion_image, (100,100))

tiger_image = cv2.resize(cv2.imread('./image/tiger.png', cv2.IMREAD_UNCHANGED), (100,100))
tiger_mask = np.full((100,100,4), 0,dtype=np.uint8)
tiger_mask[3] = tiger_image[3]
tiger_mask = cv2.resize(tiger_mask, (120,120))

jaguar_image = cv2.resize(cv2.imread('./image/jaguar.png', cv2.IMREAD_UNCHANGED), (100,100))
jaguar_mask = np.full((100,100,4), 0,dtype=np.uint8)
jaguar_mask[3] = jaguar_image[3]
jaguar_mask = cv2.resize(jaguar_mask, (120,120))

dragon_image = cv2.resize(cv2.imread('./image/dragon.png', cv2.IMREAD_UNCHANGED), (100,100))
dragon_mask = np.full((100,100,4), 0,dtype=np.uint8)
dragon_mask[3] = dragon_image[3]
dragon_mask = cv2.resize(dragon_mask, (120,120))

moles_texture = [tiger_image, jaguar_image, dragon_image]
moles_mask = [tiger_mask,jaguar_mask,dragon_mask]

seogo_logo_image = cv2.imread('./image/seogo_logo.jpg')
seogo_logo_image = cv2.resize(seogo_logo_image, (256,256))


with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        frame = cv2.flip(frame, 1)
        image = np.full((720,1280,3), 255, dtype=np.uint8)
        image[:256, 1024:] = seogo_logo_image

        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        h, w, _ = frame.shape 
        cv2.line(frame, (0, 720-height), (1280, 720-height), BLUE, 1)
        
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


            if game_start_event == False and countdown==False:
                
                cv2.line(frame, (0, 720-height), (1280, 720-height), BLUE, 1)
                
                cv2.putText(image, 'SCORE > 20',(1024, 360),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(image, ' - 1 HARIBO',(1024, 395),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(image, 'SCORE > 40',(1024, 460),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(image, ' - 2 HARIBO',(1024, 495),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

                cv2.putText(frame, 'Clap to start a Game',
                            (w//2-300, 640),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (51, 102, 153), 3, cv2.LINE_AA)
                
                # cv2.putText(frame, 'Please keep some distance or adjust your webcam',
                #             (w//2-210, h//2+210),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                # cv2.putText(frame, 'to show your whole body in camera frame',
                #             (w//2-180, h//2+230),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
                overlay(frame, (w//2-230, h//2-185), 50, 50, clap_image)
                overlay(frame, (w//2, h//2+20), 112, 210, body_image)
                
                mp_drawing.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                if get_distance(righthand, lefthand) < 30 and abs(leftz-rightz) < 20 and ( w//2-150 < nose[0] < w//2+150 and 10 < nose[1] < h//2+20):
                    countdown = True
                    start_time = time.time()
                    

            if countdown:
                count = 3 - (present_time - start_time)
                cv2.circle(frame, (w//2, h//2), 65, (0,0,0), -1)
                cv2.circle(frame, (w//2, h//2), 60, (255,255,255), -1)
                cv2.putText(frame, str(int(count)+1),
                            (w//2-20, h//2+20),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 0), 4, cv2.LINE_AA)
                if count <= 0:
                    start_time = time.time()
                    countdown = False
                    game_start_event = True
                    

            if game_start_event == True and time_remaining > 0:
                
                time_remaining = time_given - (present_time - start_time)
                ongame_ranking = ranking.head(10)[['name', 'score']]
                ongame_ranking = pd.concat([ongame_ranking, pd.DataFrame({'name' : ['player'], 'score' : [score]})])
                ongame_ranking = ongame_ranking.sort_values(by='score', ascending=False)
                if time.time() - respawn_time > respawning_time:
                    respawn_time = time.time()
                    if len(moles) + len(elites) <4:
                        if np.random.rand() < 0.1:
                            elites.append(elite_mole(random_pos()))
                        else:
                            moles.append(mole(random_pos()))
                for i in moles:
                    i.check(moles)
                for i in texts:
                    i.check(texts)
                for i in elites:
                    i.crash_detection()
                    i.move()
                    i.check(elites)
                    
                delete_idx = []
                for idx, i in enumerate(moles):
                    if (i.x-50 < righthand[0] < i.x+50 and i.y-50 < righthand[1] < i.y+50) or (i.x-50 < lefthand[0] < i.x+50 and i.y-50 < lefthand[1] < i.y+50) or (i.x-50 < rightfoot[0] < i.x+50 and i.y-50 < rightfoot[1] < i.y+50) or (i.x-50 < leftfoot[0] < i.x+50 and i.y-50 < leftfoot[1] < i.y+50):
                        overlay(frame, (i.x, i.y), 50, 50, shine_image)
                        score += 1
                        whack_sound.play()
                        #moles.append(mole(random_pos()))
                        delete_idx.append(idx)
                        texts.append(mole((i.x,i.y), 0.5,"+1"))
                delete_idx.reverse()
                for i in delete_idx:
                    del moles[i]
                    
                delete_idx = []
                for idx, i in enumerate(elites):
                    if (i.x-50 < righthand[0] < i.x+50 and i.y-50 < righthand[1] < i.y+50) or (i.x-50 < lefthand[0] < i.x+50 and i.y-50 < lefthand[1] < i.y+50) or (i.x-50 < rightfoot[0] < i.x+50 and i.y-50 < rightfoot[1] < i.y+50) or (i.x-50 < leftfoot[0] < i.x+50 and i.y-50 < leftfoot[1] < i.y+50):
                        overlay(frame, (int(i.x), int(i.y)), 50, 50, shine_image)
                        score += 3
                        whack_sound.play()
                        #moles.append(mole(random_pos()))
                        delete_idx.append(idx)
                        texts.append(mole((int(i.x),int(i.y)), 0.5,"+3"))
                delete_idx.reverse()
                for i in delete_idx:
                    del elites[i]

                cv2.putText(image, 'Score : '+str(int(score)),
                            (1024, 291),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 204), 2, cv2.LINE_AA)
                                
                cv2.putText(frame, 'Time left : '+str(int(time_remaining)),
                          (w//2+30, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 204), 2, cv2.LINE_AA)
                

                # 부채꼴 그리기
                if time_remaining > 20:
                    timer_color = GREEN
                elif time_remaining > 10:
                    timer_color = ORANGE
                else:
                    timer_color = RED
                
                angle = -90-(time_remaining/time_given)*360
                cv2.ellipse(image, (1152 ,128), (125,125), 0, angle, -90, timer_color, -1)

                #이미지 그리는 부분
                for idx, i in enumerate(ongame_ranking.iloc):
                    cv2.putText(image, ' '.join([ "%2d"%(idx+1), i['name']]),
                                (1024, 360 + idx*39),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                    cv2.putText(image, str("%3d"%i['score']),
                                (1220, 360 + idx*39),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                for i in moles:
                    overlay(frame,(i.x, i.y), 50, 50, moles_texture[i.texture])
                for i in elites:
                    overlay(frame,(int(i.x), int(i.y)), 50, 50, lion_image)
                for i in texts:
                    cv2.putText(frame, i.text, (i.x,i.y), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255), 3)
                    cv2.putText(frame,  i.text, (i.x,i.y), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,0), 2)
            
            elif game_start_event == True and time_remaining <= 0:
                
                time_remaining = 0
                game_over_event = True

        except:
            cv2.putText(frame, 'Please show your face and keep some distance from your webcam',
            (w//2-260, h//2+220),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            pass    

        # 게임종료시에만 실행
        if game_over_event == True:
            
            cv2.rectangle(frame, (w//2-170, h//2-130), (w//2+170, h//2+40), (0,0,0), -1)

            cv2.putText(frame, 'Game Over',
            (w//2-147, h//2-65),
            cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3, cv2.LINE_AA)
            
            cv2.putText(frame, 'Your Score: '+str(int(score)),
            (w//2-120, h//2),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            for idx, i in enumerate(ranking.head(10).iloc):
                cv2.putText(image, ' '.join([ "%2d"%(idx+1), i['name']]),
                            (1024, 360 + idx*39),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(image, str("%3d"%i['score']),
                            (1220, 360 + idx*39),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
            
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            if not score_recorded:
                score_board = pd.concat([score_board, pd.DataFrame({'name':['Unknown'], 'score' : [int(score)], 'time':[formatted_time]})], ignore_index=True)
                score_board.to_csv('./scoreboard.csv',index=False)
                score_board = pd.read_csv('./scoreboard.csv')
                ranking = score_board.sort_values(by='score', ascending=False)
                score_recorded = True
                
        
        image[:,:1024] = frame[:,128:1152]
        cv2.imshow('DOODEOJI', image)  # 화면크기 2배 키움
        cv2.imshow('panal', np.full((300,400,3), 255, dtype=np.uint8))
        command = cv2.waitKey(5) & 0xFF
        if command == 27:
            break
        elif command == ord('r') or command == ord('R'):
            game_start_event = False
            game_over_event = False
            game_pause_event = False
            score_recorded = False
            height = 720

            time_remaining = 99

            score = 0.1
            
        elif command == ord('q') or command == ord('Q'):
            height = 720
        elif command == ord('w') or command == ord('W'):
            height += 10
        elif command == ord('s') or command == ord('S'):
            height -= 10
cap.release()