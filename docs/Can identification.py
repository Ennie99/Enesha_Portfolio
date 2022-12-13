#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Реализация алгоритма распознавания урны на языке Python
import matplotlib.pyplot as plt
import numpy as np
import cv2
bin1 = cv2.imread('Downloads/bin10.jpg')
bin1 = cv2.cvtColor(bin1, cv2.COLOR_BGR2RGB)
plt.imshow(bin1)
plt.show()

hsv_bin1 = cv2.cvtColor(bin1, cv2.COLOR_RGB2HSV)
plt.imshow(hsv_bin1)
plt.show()

lower_green = (36,50,70)
upper_green = (70,255,255)
mask = cv2.inRange(hsv_bin1, lower_green, upper_green)
result = cv2.bitwise_and(bin1, bin1, mask=mask)
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()

blur = cv2.GaussianBlur(result, (7, 7), 0)
plt.imshow(blur)
image = cv2.imread("Desktop/f.png")
thresh = 240
im_bw = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
edges = cv2.Canny(im_bw,100,150,apertureSize = 3)
cnts = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for contour in cnts:
    approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
    cv2.drawContours(image, [approx], 0, (0, 255, 255), 3)
cv2.imshow("result", image)

