import cv2
import numpy as np
import os 

class colorSegmentation:
    def __init__(self):
        self.current_path = str.replace(os.path.abspath(os.path.curdir),"\\", "/")
        self.dataset_path =  self.current_path + "/datasets/"

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
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    color_seg = colorSegmentation()
    color_seg.diplayImage("71JvsOm1V5L._SL1000__001.jpg")