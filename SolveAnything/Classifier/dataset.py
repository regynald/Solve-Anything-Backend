import os
import sys
import django
import pickle
import imutils

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ['DJANGO_SETTINGS_MODULE'] = 'SolveAnythingBackend.settings'
django.setup()

from SolveAnythingBackend import settings
import numpy as np
import mahotas
import cv2


def load_digits(datasetPath):
    # build the dataset and then split it into data
    # and labels
    data = np.genfromtxt(datasetPath, delimiter = ",", dtype = "uint8")
    target = data[:, 0]
    data = data[:, 1:].reshape((data.shape[0], 784, 1))

    plus_train = pickle.load(open(settings.BASE_DIR + '/SolveAnything/Classifier/data/plus_training_data1.pickle', 'rb'))
    plus_data = np.array([x[0].reshape((784, 1)) for x in plus_train])
    plus_target = np.array([y[1] for y in plus_train])


    data = np.concatenate((data, plus_data))
    data = data.reshape((data.shape[0], 28, 28))
    target = np.concatenate((target, plus_target))

    # return a tuple of the data and targets
    return data, target

def deskew(image, width):
    # grab the width and height of the image and compute
    # moments for the image
    (h, w) = image.shape[:2]
    moments = cv2.moments(image)

    # deskew the image by applying an affine transformation
    skew = moments["mu11"] / moments["mu02"]
    M = np.float32([
        [1, skew, -0.5 * w * skew],
        [0, 1, 0]])
    image = cv2.warpAffine(image, M, (w, h),
        flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    # resize the image to have a constant width
    image = imutils.resize(image, width = width)

    # return the deskewed image
    return image

def center_extent(image, size):
    # grab the extent width and height
    (eW, eH) = size

    # handle when the width is greater than the height
    if image.shape[1] > image.shape[0]:
        image = imutils.resize(image, width = eW)

    # otherwise, the height is greater than the width
    else:
        image = imutils.resize(image, height = eH)

    # allocate memory for the extent of the image and
    # grab it
    extent = np.zeros((eH, eW), dtype = "uint8")
    offsetX = (eW - image.shape[1]) // 2
    offsetY = (eH - image.shape[0]) // 2
    extent[offsetY:offsetY + image.shape[0], offsetX:offsetX + image.shape[1]] = image

    # compute the center of mass of the image and then
    # move the center of mass to the center of the image
    (cY, cX) = np.round(mahotas.center_of_mass(extent)).astype("int32")
    (dX, dY) = ((size[0] // 2) - cX, (size[1] // 2) - cY)
    M = np.float32([[1, 0, dX], [0, 1, dY]])
    extent = cv2.warpAffine(extent, M, size)

    # return the extent of the image
    return extent
