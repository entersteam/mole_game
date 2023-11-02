import cv2
import numpy as np
import pandas as pd

image = np.full((720,1280,3), 255, dtype=np.uint8)

image[:,:1024,:] = 0


score_board = pd.read_csv('./scoreboard.csv')
ranking = score_board.sort_values(by='score', ascending=False)
for idx, i in enumerate(ranking.head(10).iloc):
    cv2.putText(image, ' '.join([ "%2d"%(idx+1), i['name']]),
                (1024, 360 + idx*39),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(image, str("%3d"%i['score']),
                (1220, 360 + idx*39),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    
cv2.imshow('test', image)
cv2.waitKey()