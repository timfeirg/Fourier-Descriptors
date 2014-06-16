#!/usr/local/bin/python

import cv2
import numpy
import matplotlib.pyplot as plt


def findDescriptor(img):
    contour = []
    contour, hierarchy = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
        contour)
    contour_array = contour[0][:, 0, :]
    contour_complex = numpy.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]
    fourier_result = numpy.fft.fft(contour_complex)
    magnitude = numpy.absolute(fourier_result)
    return fourier_result, contour


def reconstruct(dscptr, degree):
    """ reconstruct(dscptr, degree) attempts to reconstruct the image
    using the first [degree] descriptors of dscptr"""
    dscptr = numpy.fft.fftshift(dscptr)
    plt.subplot(211)
    plt.plot(numpy.absolute(dscptr))
    center_index = len(dscptr)/2
    print(center_index)
    descriptor_in_use = dscptr[center_index - degree:center_index + degree]
    plt.subplot(212)
    plt.plot(numpy.absolute(descriptor_in_use))
    plt.show()
    contour_reconstruct = numpy.fft.ifft(descriptor_in_use)
    pass

black = numpy.zeros((500, 500), numpy.uint8)
star = cv2.imread("/Users/timfeirg/Documents/Fourier-Descriptor/star.jpg", 0)
retval, star = cv2.threshold(star, 127, 255, cv2.THRESH_BINARY)
fourier_result, contour = findDescriptor(star)
contour_reconstruct = reconstruct(fourier_result, 50)
