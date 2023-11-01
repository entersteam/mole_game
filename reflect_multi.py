import numpy as np
import cv2

def unit_vector(vector):
    unit = vector / np.linalg.norm(vector)
    return unit

class ball:    
    def __init__(self,speed=5) -> None:
        self.speed = speed
        self.vector = unit_vector(np.random.normal(0, 1, 2))
        self.pos = np.array([200,200], dtype=np.float64)
            
    def reflect(self, norm):
        self.vector += 2 * norm * np.dot(norm,-self.vector)
        
    def crash_detection(self):
        if self.pos[0]>400:
            norm = np.array((-1,0))
            self.reflect(norm)
        if self.pos[0]<0:
            norm = np.array((1,0))
            self.reflect(norm)
        if self.pos[1]>400:
            norm = np.array((0,1))
            self.reflect(norm)
        if self.pos[1]<0:
            norm = np.array((0,-1))
            self.reflect(norm)
        
    def move(self):
        self.pos += self.vector*self.speed
        

balls = [ball()]

def mouse_event(event, x, y, flags, param):
    global balls
    if event == cv2.EVENT_FLAG_LBUTTON:
        balls.append(ball(np.random.randint(3,7)))

    elif event == cv2.EVENT_FLAG_RBUTTON:
        balls.pop(0)

window = cv2.namedWindow('geometry')
cv2.setMouseCallback('geometry', mouse_event)

while True:
    scr = np.full((400,400,3),(255,255,255), dtype=np.uint8)
    for i in balls:
        cv2.circle(scr, i.pos.astype(int), 3, (0,0,0))
        i.crash_detection()
        i.move()
    cv2.imshow('geometry', scr)
    command = cv2.waitKey(16) & 0xFF
    if  command == 27:
        break
cv2.destroyAllWindows