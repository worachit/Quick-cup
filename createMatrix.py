import os 
import numpy as np
import cv2

import glob
import sys

__author__ = "Tiziano Fiorenzani"
__date__ = "01/06/2018"


class createMatrix:
    def __init__(self, camera_number = 0):
        self.CURRENT_PATH = str.replace(os.path.dirname(os.path.realpath(__file__)),"\\", "/")
        
        self.SNAPSHOT_PATH = self.CURRENT_PATH + "/CalibrationSnapShot/CameraMatrix/"
        self.MATRIX_PATH =  self.CURRENT_PATH + "/Matrix/"

        self.camera_number = camera_number

        self.VIDEO_RESOLUTION = (1280, 720)

        self.DISTORT_MATRIX_NAME = "cameraDistortion"
        self.CAMERA_MATRIX_NAME = "cameraMatrix"
        self.PERSPECTIVE_MATRIX_NAME = "perspectiveMatrix"

        self.camera_matrix = None
        self.distortion_matrix = None
        self.perspective_matrix = None

    def takeImage(self):
        cap = cv2.VideoCapture(self.camera_number)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.VIDEO_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.VIDEO_RESOLUTION[1])

        nSnap   = 0
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        fileName    = "%s_%d_%d_" %("snapshot", w, h)
        while True:
            _ , frame = cap.read()

            cv2.imshow('camera', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                print("Saving image ", nSnap)
                cv2.imwrite(self.SNAPSHOT_PATH + "%s%d.jpg"%(fileName, nSnap), frame)
                nSnap += 1

        cap.release()
        cv2.destroyAllWindows()

        print("Files saved")

    def createCamMatrix(self):
        nRows = 9
        nCols = 7
        dimension = 20 #- mm

        workingFolder = self.CURRENT_PATH + "/CalibrationSnapShot/CameraMatrix/"
        imageType = 'jpg'

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

        objp = np.zeros((nRows*nCols,3), np.float32)
        objp[:,:2] = np.mgrid[0:nCols,0:nRows].T.reshape(-1,2)

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        filename = workingFolder + "/*." + imageType
        images = glob.glob(filename)

        if len(images) < 9:
            print("Not enough images were found: at least 9 shall be provided!!!")
            sys.exit()
        
        else:
            nPatternFound = 0
            imgNotGood = images[1]

            for fname in images:
                if 'calibresult' in fname: continue
                #-- Read the file and convert in greyscale
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                print("Reading image ", fname)

                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, (nCols,nRows),None)

                # If found, add object points, image points (after refining them)
                if ret == True:
                    print("Pattern found! Press ESC to skip or ENTER to accept")
                    #--- Sometimes, Harris cornes fails with crappy pictures, so
                    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (nCols,nRows), corners2,ret)
                    cv2.imshow('img',img)
                    k = cv2.waitKey(0) & 0xFF
                    if k == 27: #-- ESC Button
                        print("Image Skipped")
                        imgNotGood = fname
                        continue

                    print("Image accepted")
                    nPatternFound += 1
                    objpoints.append(objp)
                    imgpoints.append(corners2)

                else:
                    imgNotGood = fname

        cv2.destroyAllWindows()
        
        if (nPatternFound > 1):
            print("Found %d good images" % (nPatternFound))
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

            self.camera_matrix = mtx
            self.distortion_matrix = dist

            img = cv2.imread(imgNotGood)
            new_img = self.applyCameraMatrix(img)

            cv2.imwrite(workingFolder + "/calibresult.png",new_img)
            print("Calibrated picture saved as calibresult.png")
            print("Calibration Matrix: ")
            print(mtx)
            print("Disortion: ", dist)

            #--------- Save result
            filename = self.MATRIX_PATH + "/" + self.CAMERA_MATRIX_NAME + ".txt"
            np.savetxt(filename, mtx, delimiter=',')
            filename = self.MATRIX_PATH + "/" + self.DISTORT_MATRIX_NAME + ".txt"
            np.savetxt(filename, dist, delimiter=',')

            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                mean_error += error

            print("total error: ", mean_error/len(objpoints))

        else:
            print("In order to calibrate you need at least 9 good pictures... try again")

    def applyCameraMatrix(self, img):
        h, w = img.shape[:2]

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_matrix, (w,h), 1, (w,h))

        mapx,mapy = cv2.initUndistortRectifyMap(self.camera_matrix, self.distortion_matrix, None, newcameramtx, (w,h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        x,y,w,h = roi
        new_image = dst[y:y+h, x:x+w]

        return new_image

    def loadCameraMatrix(self):
        self.camera_matrix = np.loadtxt(self.MATRIX_PATH + "cameraMatrix.txt", dtype='f', delimiter=',')
        self.distortion_matrix = np.array([np.loadtxt(self.MATRIX_PATH + "cameraDistortion.txt", dtype='f', delimiter=',')])

    def createPerspectiveMatrix(self):
        cap = cv2.VideoCapture(self.camera_number)
        self.loadCameraMatrix()

        while True:
            _, frame = cap.read()
            new_frame = self.applyCameraMatrix(frame)

            cv2.circle(new_frame, (140, 230), 5, (0, 0, 255), -1)
            cv2.circle(new_frame, (451, 225), 5, (0, 0, 255), -1)
            cv2.circle(new_frame, (30, 382), 5, (0, 0, 255), -1)
            cv2.circle(new_frame, (518, 379), 5, (0, 0, 255), -1)
            
            pts1 = np.float32([[140, 230], [451, 225], [30, 382], [518, 379]])
            pts2 = np.float32([[0, 0], [600, 0], [0, 500], [600, 500]])
            
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            result = cv2.warpPerspective(new_frame, matrix, (600, 500))
            
            cv2.imshow("Frame", new_frame)
            cv2.imshow("Perspective transformation", result)
            
            key = cv2.waitKey(1)

            if key == ord('p'):
                filename = self.MATRIX_PATH + "/" + self.PERSPECTIVE_MATRIX_NAME + ".txt"
                np.savetxt(filename, matrix, delimiter=',')
                break
            
            if key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cm = createMatrix(0)
    # img = cv2.imread("snapshot_1280_720_84.jpg",1)
    # cm.loadCameraMatrix()
    # # print()
    # cv2.imshow("te1st",img)
    # cv2.imshow("test",cm.applyCameraMatrix(img))
    # # cv2.imshow("test",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # cm.takeImage()
    # cm.takeImage()
    # cm.createCamMatrix()
    cm.createPerspectiveMatrix()

    # perspective_matrix = 
    # [[2.154779969650999050e+00,8.194233687405229061e-01,-4.323216995447680802e+02],
    # [2.942091015256664832e-15,4.370257966616110323e+00,-5.244309559939340488e+02],
    # [4.445228907190568179e-18,3.338391502276205885e-03,1.000000000000000000e+00]]