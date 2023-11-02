import cv2

img = cv2.imread('./image.png')

cv2.imshow('test', img)
for i in range(8):
    print(cv2.waitKey())