#!/usr/local/bin/python

import cv2
import numpy
from FourierDescriptor import addNoise


def preprocess(img):
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    print(img)
    cv2.imshow("temp",img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    pass

plane1 = cv2.imread(
    "/Users/timfeirg/Documents/Fourier-Descriptor/plane1.png",
    0)
#plane2 = cv2.imread(
    #"/Users/timfeirg/Documents/Fourier-Descriptor/plane2.png",
    #0)

preprocess(plane1)
cv2.imshow("plane1", plane1)
cv2.waitKey(1000)
cv2.destroyAllWindows()
#contour = cv2.findContours(plane1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
