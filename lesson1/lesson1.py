import cv2
import numpy as np

#2. Please change image color through YUV space

img_bgr = cv2.imread('lenna.jpg')
cv2.imshow('img', img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
y, u, v = cv2.split(img_yuv)

y[y > 200] = 255
y[y <= 200] = y[y <= 200]+55

u[u > 100] = 255
u[u <= 100] = u[u <= 100]+155

v[v > 150] = 255
v[v <= 150] = v[v <= 150]+105

img_yuv_new=cv2.merge((y, u, v))
img_bgr_new=cv2.cvtColor(img_yuv_new, cv2.COLOR_YUV2BGR)
cv2.imshow('new img', img_bgr_new)
cv2.waitKey(0)
cv2.destroyAllWindows()

#3 Combine image crop, color shift, rotation and perspective transform together to complete a data augmentation script

