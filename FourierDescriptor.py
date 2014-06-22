#!/usr/local/bin/python

import cv2
import numpy

MIN_DESCRIPTOR = 8  # surprisingly enough, 2 descriptors are already enough
TRAINING_SIZE = 100


def findDescriptor(img):
    """ findDescriptor(img) finds and returns the
    Fourier-Descriptor of the image contour"""
    contour = []
    contour, hierarchy = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
        contour)
    #print('contour report: length = ', len(contour), 'to ', len(contour[0]))
    black = numpy.zeros((800, 800), numpy.uint8)
    cv2.drawContours(black, contour, -1, 255, thickness=-1)
    #cv2.imshow("black", black)
    # cv2.waitKey(500)
    # cv2.destroyAllWindows()
    contour_array = contour[0][:, 0, :]
    contour_complex = numpy.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]
    fourier_result = numpy.fft.fft(contour_complex)
    #print('descriptors report: length = ', len(fourier_result), type(fourier_result))
    return fourier_result


def truncate_descriptor(descriptors, degree):
    center_index = len(descriptors)/2 - 1
    descriptors = descriptors[
        center_index - degree / 2:center_index + degree / 2]
    return descriptors


def reconstruct(descriptors, degree):
    """ reconstruct(descriptors, degree) attempts to reconstruct the image
    using the first [degree] descriptors of descriptors"""
    descriptors = numpy.fft.fftshift(descriptors)
    descriptors = truncate_descriptor(descriptors, degree)

    descriptor_in_use = numpy.fft.ifftshift(descriptor_in_use)
    contour_reconstruct = numpy.fft.ifft(descriptor_in_use)
    contour_reconstruct = numpy.array(
        [contour_reconstruct.real, contour_reconstruct.imag])
    contour_reconstruct = numpy.transpose(contour_reconstruct)
    contour_reconstruct = numpy.expand_dims(contour_reconstruct, axis=1)
    # normalization
    contour_reconstruct *= 500 / contour_reconstruct.max()
    # type cast to int32
    contour_reconstruct = contour_reconstruct.astype(numpy.int32, copy=False)
    black = numpy.zeros((800, 800), numpy.uint8)

    cv2.drawContours(black, contour_reconstruct, -1, 255, thickness=-1)
    return descriptor_in_use


def addNoise(descriptors):
    """this function adds gaussian noise to descriptors
    descriptors should be a [N,2] numpy array"""
    scale = descriptors.max() / 10
    noise = numpy.random.normal(0, scale, descriptors.shape[0])
    noise = noise + 1j * noise
    descriptors += noise


def sample_generater(sample1, sample2):
    """this function generates training_set, also for testing"""
    response = numpy.array([0, 1])
    response = numpy.tile(response, TRAINING_SIZE / 2)
    response = response.astype(numpy.float32)
    training_set = numpy.empty(
        [TRAINING_SIZE, MIN_DESCRIPTOR], dtype=numpy.float32)
    # assign descriptors with noise to our training_set
    for i in range(0, TRAINING_SIZE - 1, 2):
        descriptors_sample1 = findDescriptor(sample1)
        descriptors_sample1 = truncate_descriptor(
            descriptors_sample1,
            MIN_DESCRIPTOR)
        addNoise(descriptors_sample1)
        training_set[i] = numpy.absolute(descriptors_sample1)
        descriptors_sample2 = findDescriptor(sample2)
        descriptors_sample2 = truncate_descriptor(
            descriptors_sample2,
            MIN_DESCRIPTOR)
        addNoise(descriptors_sample2)
        training_set[i + 1] = numpy.absolute(descriptors_sample2)
    return training_set, response

"""Descriptor"""
#src = cv2.imread("/Users/timfeirg/Documents/Fourier-Descriptor/licoln.tif", 0)
#retval, src = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY)
#fourier_result, contour = findDescriptor(src)
#contour_reconstruct = reconstruct(fourier_result, 20)
"""Descriptor END"""

"""generate training_set"""
# import images and treat
sample1 = cv2.imread(
    "/Users/timfeirg/Documents/Fourier-Descriptor/plane1.tiff",
    0)
sample2 = cv2.imread(
    "/Users/timfeirg/Documents/Fourier-Descriptor/plane2.tiff",
    0)
# prepocess
retval, sample1 = cv2.threshold(sample1, 127, 255, cv2.THRESH_BINARY_INV)
retval, sample2 = cv2.threshold(sample2, 127, 255, cv2.THRESH_BINARY_INV)
del retval  # useless
training_set, response = sample_generater(sample1, sample2)
test_set, correct_answer = sample_generater(sample1, sample2)
"""generate training_set END"""

"""SVM START"""
#"""Training!"""
#svm_model = cv2.SVM()
#svm_model.train(training_set, response, params=svm_params)

# set up parameters for SVM
 # svm_params = dict(
     # kernel_type=cv2.SVM_LINEAR,
     # svm_type=cv2.SVM_C_SVC,
     # C=1
#)

#"""Guessing part, to my surprise SVM training is already perfect with
# 2 descriptors"""
#answer_SVM = [svm_model.predict(s) for s in test_set]
#answer_SVM = numpy.array(answer_SVM)
#error_rate_SVM = numpy.sum(numpy.in1d(correct_answer, answer_SVM)) / TRAINING_SIZE
#print('For SVM, error rate (0~1) = ', error_rate_SVM)
"""SVM END"""

"""Minimum distance classifier"""
#k_nearest = cv2.KNearest(training_set, response)
#ret, answer_KNN, neignbours, distance = k_nearest.find_nearest(training_set, 3)
#error_rate_KNN = numpy.sum(numpy.in1d(correct_answer, answer_KNN)) / TRAINING_SIZE
#print('For KNN, error_rate_KNN = ', error_rate_KNN)
"""Minimum distance classifier END"""

"""Bayers classifier"""
bayers_model = cv2.NormalBayesClassifier()
bayers_model.train(training_set, response)
retval, answer_bayers = bayers_model.predict(test_set)
error_rate_bayers = numpy.sum(
    numpy.in1d(
        correct_answer,
        answer_bayers)) / TRAINING_SIZE
print('For bayers_model, error_rate_bayers = ', error_rate_bayers)
"""Bayers classifier END"""
