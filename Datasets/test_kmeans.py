# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import cv2


# img = cv2.imread("71JvsOm1V5L._SL1000__016.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# r, g, b = cv2.split(img)
# r = r.flatten()
# g = g.flatten()
# b = b.flatten()
# #plotting 
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(r, g, b)
# plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt

original_image = cv2.imread("71JvsOm1V5L._SL1000__001.jpg")

# alpha = 1.5
# beta = 50


# cv2.imshow("test",result)
# cv2.waitKey(0)



img = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)

# img = cv2.addWeighted(_img,alpha, np.zeros(_img.shape, _img.dtype), 0, beta)


vectorized = img.reshape((-1,3))

vectorized = np.float32(vectorized)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 6
attempts = 20
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

res = center[label.flatten()]
result_image = res.reshape((img.shape))

figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()
