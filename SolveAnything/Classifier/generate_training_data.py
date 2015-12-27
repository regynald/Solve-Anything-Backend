import os
import sys
import django
import cv2
import mahotas
import numpy as np
import cPickle as pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ['DJANGO_SETTINGS_MODULE'] = 'SolveAnythingBackend.settings'
django.setup()

from SolveAnything.models import Problem
from SolveAnything.Classifier import dataset

training_samples = []

def is_inner_rectangle(inner, rectangles):
    #return largest containing rectangle
    for outer in rectangles:
        if inner != outer:
            x1, y1, w1, h1 = outer
            x2, y2, w2, h2 = inner
            if x1 < x2 and y1 < y2 and x1 + w1 > x2 + w2 and y1 + h1 > y2 + h2:
                return True
    return False

def smallest_bounding_rectangle(rectangle, rectangles, leeway = 5):
    #return smallest bounding rectangle
    smallest_area = sys.maxint
    smallest_bounding = None
    overlapping = False
    for other_rectangle in rectangles:
        if rectangle != other_rectangle:
            currently_overlapping = True
            x1, y1, w1, h1 = rectangle # x1, y1 is top left
            x2, y2, w2, h2 = other_rectangle # x2, y2 is top left

            x3, y3 = x1 + w1, y1 + h1 # x3, y3 is bottom right of rect
            x4, y4 = x2 + w2, y2 + h2 # x4, y4 is bottom right of other rect

            if x1 - leeway > x4 + leeway or x2 - leeway > x3 + leeway:
                currently_overlapping = False
            if y1 - leeway > y4 + leeway or y2 - leeway > y3 + leeway:
                currently_overlapping = False
            if currently_overlapping:
                min_x, min_y, max_x, max_y = min(x1, x2), min(y1, y2), max(x3, x4), max(y3, y4)
                bounding_rectangle = (min_x, min_y, max_x - min_x, max_y - min_y)
                area = abs((max_x - min_y) * (max_y - min_y))

                if area < smallest_area:
                    smallest_area, smallest_bounding, overlapping = area, bounding_rectangle, True
    return overlapping, smallest_bounding

def valid_rectangles(rectangles, leeway = 5):
    valid = {}

    for rectangle in rectangles:
        is_overlapping, bounding_rectangle = smallest_bounding_rectangle(rectangle, rectangles, leeway)
        is_valid_rectangle = True
        valid_rectangle = rectangle
        if is_inner_rectangle(rectangle, rectangles):
            is_valid_rectangle = False
        if is_overlapping:
            valid_rectangle = bounding_rectangle
        if is_valid_rectangle:
            valid[valid_rectangle] = 0

    rectangles = valid.keys()

    bad_rectangles = {}
    for rectangle in rectangles:
        is_overlapping, bounding_rectangle = smallest_bounding_rectangle(rectangle, rectangles, leeway)
        if is_inner_rectangle(rectangle, rectangles):
            bad_rectangles[rectangle] = 0
        if is_overlapping:
            bad_rectangles[rectangle] = 0

    rectangles.sort(key=lambda x: x[0])
    return rectangles, len(bad_rectangles.keys())

def generate_training_data(problem_id, exclusions, threshold, leeway, transformations, label):
    try:
        problem = Problem.objects.get(id=problem_id)
    except Problem.DoesNotExist:
        return None

    global nn, f_output

    im = cv2.imread(problem.image.path)

    im_height, im_width = im.shape[:2]

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    im_blurred = cv2.GaussianBlur(im_gray, (9, 9), 0)

    im_edged = cv2.Canny(im_blurred, 30, 150)

    (im_contoured, contours, _) = cv2.findContours(im_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted([(contour, cv2.boundingRect(contour)[0]) for contour in contours], key =lambda x: x[1])

    rectangles = [cv2.boundingRect(contour[0]) for contour in contours]

    leeway = leeway
    if leeway == None:
        leeway = 0

    rectangles, bad_rectangles = valid_rectangles(rectangles, leeway)

    while bad_rectangles > 0:
        rectangles, bad_rectangles = valid_rectangles(rectangles, leeway)

    i = 0
    for rectangle in rectangles:
        x, y, w, h = rectangle

        if 7 <= w <= im_width / 4 and 20 <= h <= im_height / 4:
            if i not in exclusions:
                roi = im_gray[y: y + h, x: x + w]
                im_thresh = roi.copy()
                if threshold == None:
                    threshold = mahotas.thresholding.otsu(roi)
                im_thresh[im_thresh > threshold] = 255
                im_thresh = cv2.bitwise_not(im_thresh)

                im_thresh = dataset.center_extent(im_thresh, (28, 28))

                training_sample = im_thresh.copy()
                training_samples.append((training_sample.reshape((784, 1)), label))
                if 'rot90' in transformations:
                    training_samples.append((np.rot90(training_sample, 3).reshape((784, 1)), label))
                if 'rot180' in transformations:
                    training_samples.append((np.rot90(training_sample, 2).reshape((784, 1)), label))
                if 'rot270' in transformations:
                    training_samples.append((np.rot90(training_sample, 1).reshape((784, 1)), label))
                if 'fliplr' in transformations:
                    training_samples.append((np.fliplr(training_sample).reshape((784, 1)), label))
                if 'flipud' in transformations:
                    training_samples.append((np.flipud(training_sample).reshape((784, 1)), label))

                # print i
                # cv2.imshow('roi', roi)
                # cv2.imshow('thresh', im_thresh)
                # cv2.waitKey(0)

                color = (0, 0, 255)

                cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)
            i += 1

    print problem_id, i

    # cv2.imshow('image', im)
    # cv2.waitKey(0)
    # print '-' * 25


def save_data(filename = 'data/four_operators.pickle'):
    with open(filename, 'wb') as output:
        pickle.dump(training_samples, output, pickle.HIGHEST_PROTOCOL)


sample_inputs = range(372, 389)

sample_addition = [374, 378, 381, 382, 386, 387, 289, 290, 291, 292, 293, 294, 295]
bad_addition = [[], [2, 5, 40, ], [11], [], [54], [], [], [], [], [], [], [], [],]
addition_transformations = ['rot90', 'rot180', 'rot270', 'fliplr', 'flipud', ]

sample_subtraction = [377, 379, ]
bad_subtraction = [[], [], ]
subtraction_transformations = ['rot180', 'fliplr', 'flipud', ]

sample_multiplication = [372, 380, 383, 384, 388, ]
bad_multiplication = [[], [], [], [15], [], ]
multiplication_transformations = ['rot90', 'rot180', 'rot270', 'fliplr', 'flipud', ]

sample_division = [373, 375, 376, 385, ]
bad_division = [[], [1], [], [], ]
division_transformations = ['rot180', ]

samples = [(sample_addition, bad_addition, None, None, addition_transformations, 10),
           (sample_subtraction, bad_subtraction, None, None, subtraction_transformations,  11),
           (sample_multiplication, bad_multiplication, None, 5, multiplication_transformations, 12),
           (sample_division, bad_division, 127, 25, division_transformations, 13)]

for inputs, exclusions, threshold, leeway, supported_transformations, label in samples:
    for i in range(len(inputs)):
        input = inputs[i]
        exclusions_for_input = exclusions[i]
        generate_training_data(input, exclusions_for_input, threshold, leeway, supported_transformations, label)

save_data()
