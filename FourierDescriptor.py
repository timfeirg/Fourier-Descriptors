#!/usr/local/bin/python

import cv2
import numpy
import matplotlib.pyplot as plt


def findDescriptor(img):
    """ findDescriptor(img) finds and returns the 
    Fourier-Descriptor of the image contour"""
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


def reconstruct(descriptors, degree):
    """ reconstruct(descriptors, degree) attempts to reconstruct the image
    using the first [degree] descriptors of descriptors"""
    descriptors = numpy.fft.fftshift(descriptors)

    # plot the descriptor in frequency domain just like in matlab
    plt.subplot(211)
    plt.plot(numpy.absolute(descriptors))
    center_index = len(descriptors)/2
    descriptor_in_use = descriptors[center_index - degree:center_index + degree]
    plt.subplot(212)
    plt.plot(numpy.absolute(descriptor_in_use))
    plt.show()

    descriptor_in_use = numpy.fft.ifftshift(descriptor_in_use)
    contour_reconstruct = numpy.fft.ifft(descriptor_in_use)
    contour_reconstruct = numpy.array(
        [contour_reconstruct.real, contour_reconstruct.imag])
    contour_reconstruct = numpy.transpose(contour_reconstruct)
    contour_reconstruct = numpy.expand_dims(contour_reconstruct, axis=1)
    return contour_reconstruct

black = numpy.zeros((800, 800), numpy.uint8)
star = cv2.imread("/Users/timfeirg/Documents/Fourier-Descriptor/star.jpg", 0)
retval, star = cv2.threshold(star, 127, 255, cv2.THRESH_BINARY)
fourier_result, contour = findDescriptor(star)
contour_reconstruct = reconstruct(fourier_result, 60)

# normalization
contour_reconstruct *= 800 / contour_reconstruct.max()
# type cast to int32
contour_reconstruct = contour_reconstruct.astype(numpy.int32, copy=False)
cv2.drawContours(black, contour_reconstruct, -1, 255)
cv2.imshow("black", black)
cv2.waitKey()
cv2.destroyAllWindows()
