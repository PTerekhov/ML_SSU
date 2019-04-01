import cv2
import matplotlib.pyplot as plt
from random import randint
import numpy as np

img = cv2.imread('money.jpg')
plt.imshow(img)

imgrbg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(imgrbg)

# переводим в grayscale
gray = cv2.cvtColor(imgrbg, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')
gray1 = cv2.GaussianBlur(gray, (5,5), 11)
ret, thresh1 = cv2.threshold(gray1,250,255,cv2.THRESH_BINARY_INV)
plt.imshow(thresh1)

kernel = np.ones((5, 5),np.uint8)
erode = cv2.erode(thresh1,kernel,iterations = 12)
plt.imshow(erode)

imgrbg, contours = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("{} contours have been found".format(len(contours)))

for contour in contours:
    cv2.drawContours(img, [contour], 0, (randint(0, 255), randint(0, 255), randint(0, 255)), -1)
plt.imshow(img)
plt.show()

