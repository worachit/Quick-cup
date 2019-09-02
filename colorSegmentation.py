import cv2
import numpy as np
import string
import os
from matplotlib import pyplot as plt

class colorSegmentation:
    def __init__(self):
        self.current_path = str.replace(os.path.dirname(os.path.realpath(__file__)),"\\", "/")
        self.dataset_path =  self.current_path + "/Datasets/"
        
        self.ranges = {
            "red" : (-15,15),
            "yellow" : (15,30),
            "green" : (40,80),
            "blue" : (100,140)
        }
        self.gray_range = (150,255) #range in gray scale, not hue 

        self.masks = {
            "red" : None,
            "yellow" : None,
            "green" : None,
            "blue" : None,
            "gray" : None
        }

        self.saturation = (50,255)
        self.value = (50,255)


    def _basicRead(self,file_name):
        img = cv2.imread(self.dataset_path + file_name,1)
        # img = cv2.resize(img, self.resize_wh)
        return img   

    def diplayImage(self, image, name = None):
        #image can be either image object or file name. 
        if type(image) is np.ndarray:
            img = image
            if name == None:
                name = "image"
        else:
            img = self._basicRead(image)
            if name == None:
                name = str(image)

        cv2.imshow(name , img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def _color_wheel(self, color):
        color_range = self.ranges[color]
        color_list = [i if i > 0 else i+180 for i in color_range]
        color_list = [[color_list[0] , 180],[0,color_list[1]]] if color_range[0] * color_range[1] < 0 else [color_list]
        return color_list

    def _averageColor(self, bgr_img):
        all_means = []

        for BGR_number in range(3):
            data = bgr_img[BGR_number]
            mean = data[np.nonzero(data)].mean()
            all_means.append(int(mean)) 

        return np.array(all_means)

    def createColorMask(self, file_name, color):
        img = self._basicRead(file_name)
        hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask = 0

        for hue in self._color_wheel(color):
            lower_color = np.array([hue[0], self.saturation[0], self.value[0]])
            upper_color = np.array([hue[1], self.saturation[1], self.value[1]])
            mask_sep = cv2.inRange(hsv_img, lower_color, upper_color)
            
            mask += mask_sep

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((2,2), np.uint8))

        maskRGB = cv2.bitwise_and(img,img,mask=mask)
        
        self.masks[color] = mask

        return (mask, maskRGB)

    def createGrayMask(self, file_name):
        img = self._basicRead(file_name)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        for color in self.ranges:
            gray = cv2.bitwise_or(gray, self.masks[color])
        
        mask = cv2.inRange(gray, self.gray_range[0] , self.gray_range[1])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))
        mask = cv2.bitwise_not(mask)

        maskRGB = cv2.bitwise_and(img,img,mask=mask)

        self.masks["gray"] = mask

        return (mask, maskRGB)

    def createAllMask(self,image):
        blue , _ = self.createColorMask(image, "blue")
        red , _ = self.createColorMask(image, "red")
        green , _ = self.createColorMask(image, "green")
        yellow , _ = self.createColorMask(image, "yellow")
        gray, _ = self.createGrayMask(image)

        return (blue, green, red, yellow, gray)

    def showAllMask(self, image = None):
        if not all(self.masks.values()) or image == None:
            all_mask = np.hstack(self.createAllMask(image))
        else:
            masks = [self.masks[i] for i in self.masks]
            all_mask = np.hstack(masks)
        cv2.imshow('mask', all_mask)


if __name__ == "__main__":
    color_seg = colorSegmentation()

    image = "71JvsOm1V5L._SL1000__015.jpg"

    color_seg.showAllMask(image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()