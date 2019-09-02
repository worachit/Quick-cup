import cv2
import numpy as np


img = cv2.imread('2019-08-22_07-48-48.jpg',1)

camera_matrix = np.loadtxt("matrix/cameraMatrix.txt", dtype='f', delimiter=',')
distortion_matrix = np.array([np.loadtxt("matrix/cameraDistortion.txt", dtype='f', delimiter=',')])

# print(camera_matrix)
# print(distortion_matrix)


# img = cv2.imread(imgNotGood)
h,  w = img.shape[:2]
# print("Image to undistort: ", imgNotGood)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_matrix, (w,h), 1, (w,h))

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(camera_matrix,distortion_matrix,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
new_image = dst[y:y+h, x:x+w]

cv2.imshow('image',new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()