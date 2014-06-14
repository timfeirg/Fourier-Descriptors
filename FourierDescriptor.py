#!/usr/local/bin/python

import cv2
import numpy
import matplotlib.pyplot


def findDescriptor(img):
    contour = []
    contour, hierarchy = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
        contour)
    contour_array = contour[0][:, 0, :]
    contour_complex = contour_array[:, 0]
    contour_complex += 1j*contour_array[:, 1]
    # time to conduct some fourier transform
    fourier_result = numpy.fft.fft(contour_complex)
    return fourier_result


def reconstruct(dscptr, degree):
    """ reconstruct(dscptr, degree) attempts to reconstruct the image
    using the first [degree] descriptors of dscptr"""


cv2.destroyAllWindows()

# create a circle for testing
star = cv2.imread("/Users/timfeirg/Documents/Fourier-Descriptor/star.jpg", 0)
retval, star = cv2.threshold(star, 127, 255, cv2.THRESH_BINARY)
fourier_result = findDescriptor(star)

# reconstruct using certain amount of descriptors
reconstruct(fourier_result, 3)
