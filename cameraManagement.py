import os 
import numpy as np
import cv2

import time
import datetime


class cameraManagement:
    def __init__(self, camera_number = 0):
        self.CURRENT_PATH = str.replace(os.path.dirname(os.path.realpath(__file__)),"\\", "/")
        self.MATRIX_PATH =  self.CURRENT_PATH + "/Matrix/"
        self.VIDEO_CAPTURE_PATH = self.CURRENT_PATH + "/VideoCapture/"

        self.camera_number = camera_number

        self.VIDEO_RESOLUTION = (1280, 720)

        self.main_image = None

        self.camera_matrix = None
        self.distortion_matrix = None
        self.perspective_matrix = None
    
    def loadMatrix(self):
        self.camera_matrix = np.loadtxt(self.MATRIX_PATH + "cameraMatrix.txt", dtype='f', delimiter=',')
        self.distortion_matrix = np.array([np.loadtxt(self.MATRIX_PATH + "cameraDistortion.txt", dtype='f', delimiter=',')])
        self.perspective_matrix = np.loadtxt(self.MATRIX_PATH + "perspectiveMatrix.txt", dtype='f', delimiter=',')
        return True 

    def displayMatrix(self):
        print("camera matrix : \n{}".format(self.camera_matrix))
        print("\n distortion matrix : \n{}".format(self.distortion_matrix))
        print("\n perspective matrix : \n{}".format(self.perspective_matrix))

    def openVideo(self, **kwargs):
        apply_matrix = kwargs.get('apply_matrix', True)

        self.loadMatrix()
        cap = cv2.VideoCapture(self.camera_number)
        cap.set(3, self.VIDEO_RESOLUTION[0])
        cap.set(4, self.VIDEO_RESOLUTION[1])
        
        while True:
            _, frame = cap.read()
            
            if apply_matrix:
                frame = self.applyMatrix(frame)
            
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def captureImage(self, **kwargs):
        number_of_image = kwargs.get('number_of_image', 1)
        delay = kwargs.get('delay', 5000.0) #millis
        apply_matrix = kwargs.get('apply_matrix', True)

        self.loadMatrix()
        cap = cv2.VideoCapture(self.camera_number)
        cap.set(3, self.VIDEO_RESOLUTION[0])
        cap.set(4, self.VIDEO_RESOLUTION[1])
        
        count = 0
        
        start_time = time.time()*1000.0

        while True:
            present_time = time.time()*1000.0

            success, frame = cap.read()
            if apply_matrix:
                frame = self.applyMatrix(frame)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)

            if count >= number_of_image:
                break
            
            if success and present_time - start_time > delay:
                start_time = time.time()*1000.0

                print("take image No. %d" % count)
                cv2.imwrite(self.VIDEO_CAPTURE_PATH + "snapshot-%dx%d-%d.png"%(self.VIDEO_RESOLUTION[0], self.VIDEO_RESOLUTION[1], count), frame)
                count += 1

            if key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("succesfully capture images")

        # need to adjust the return value to surve multiple output image
        return frame

    def createMainImage(self, image = None):
        self.main_image = self.captureImage() if image == None else image

    def _applyCameraMatrix(self, img):
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_matrix, (w,h), 1, (w,h))

        mapx,mapy = cv2.initUndistortRectifyMap(self.camera_matrix, self.distortion_matrix, None, newcameramtx, (w,h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        x,y,w,h = roi
        new_image = dst[y:y+h, x:x+w]

        self.main_image = new_image
        
        return new_image

    def applyMatrix(self, img , **kwargs):
        apply_perspective = kwargs.get('apply_perspective', True)
        displayImage = kwargs.get('apply_perspective', False)
        
        img = self._applyCameraMatrix(img)

        if apply_perspective:
            img = cv2.warpPerspective(img, self.perspective_matrix, (500, 600))
        
        if displayImage:
            print("succesfully transform images")
            cv2.imshow('image',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img


if __name__ == "__main__":
    cam_man = cameraManagement()
    cam_man.loadMatrix()
    cam_man.openVideo()






