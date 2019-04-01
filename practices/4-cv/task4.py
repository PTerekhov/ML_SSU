import cv2
import matplotlib.pyplot as plt
from random import randint
import numpy as np

img = cv2.imread('money.jpg')
plt.imshow(img)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)

# переводим в grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
plt.imshow(img_gray, cmap='gray')
img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 11)
ret, thresh1 = cv2.threshold(img_gray_blur, 250, 255, cv2.THRESH_BINARY_INV)
plt.imshow(thresh1)

kernel = np.ones((5, 5),np.uint8)
erode = cv2.erode(thresh1,kernel,iterations = 12)
plt.imshow(erode)

img_rgb, contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("{} contours have been found".format(len(contours)))

for contour in contours:
    cv2.drawContours(img, [contour], 0, (randint(0, 255), randint(0, 255), randint(0, 255)), -1)
plt.imshow(img)
plt.show()

