import cv2
import numpy as np
import pandas as pd

image = np.full((720,1280,3), 255, dtype=np.uint8)

image[:,:1024,:] = 0


score_board = pd.read_csv('./scoreboard.csv')
ranking = score_board.sort_values(by='score', ascending=False)


cv2.putText(image, 'SCORE > 20',(1024, 360),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
cv2.putText(image, ' - 1 HARIBO',(1024, 395),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
cv2.putText(image, 'SCORE > 40',(1024, 460),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
cv2.putText(image, ' - 2 HARIBO',(1024, 495),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    
cv2.imshow('test', image)
cv2.waitKey()