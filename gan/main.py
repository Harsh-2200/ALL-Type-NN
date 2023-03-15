import cv2 
import time


img = cv2.imread('name.png')

height , width = img.shape[0:2]

img = cv2.resize(img , ( height // 2  , width // 2 ))
height , width = img.shape[0:2]

cv2.imshow('img' , img)


super_res = cv2.dnn_superres.DnnSuperResImpl_create()
super_res.readModel('LapSR_x4.pb')
super_res.setModel('lapsrn',4)
lap_image = super_res.upsample(img)

cv2.imshow('lap_image' , lap_image)










cv2.waitKey(0)
cv2.destroyAllWindows()
