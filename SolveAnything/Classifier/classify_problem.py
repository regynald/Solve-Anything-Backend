import os
import sys
import django
import cv2
import mahotas
import imutils

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ['DJANGO_SETTINGS_MODULE'] = 'SolveAnythingBackend.settings'
django.setup()

from imutils import object_detection
from django.core.files import File
from SolveAnything.models import Problem
from SolveAnything.Classifier import dataset
from SolveAnything.Classifier.CNN.cnn import loadNeuralNetworkFromFile
from SolveAnything.Classifier.SVM.svm import loadSupportVectorMachineFromFile


possible_values = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/']

nn, f_output = loadNeuralNetworkFromFile(filename='shallow-cnn-solve-anything-model-01-27-16.pkl')
svm, hog = loadSupportVectorMachineFromFile(filename='svm-linear-SVC-solve-anything-model-01-21-2016')


def is_inner_rectangle(inner, rectangles):
    """
    return largest containing rectangle
    """
    for outer in rectangles:
        if inner != outer:
            x1, y1, w1, h1 = outer
            x2, y2, w2, h2 = inner
            if x1 < x2 and y1 < y2 and x1 + w1 > x2 + w2 and y1 + h1 > y2 + h2:
                return True
    return False


def smallest_bounding_rectangle(rectangle, rectangles, leeway = 5):
    """
    return smallest bounding rectangle
    """
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


def classify_problem(problem_id):
    try:
        problem = Problem.objects.get(id=problem_id)
    except Problem.DoesNotExist:
        problem = None
        return []

    global nn, f_output

    im = cv2.imread(problem.image.path)

    im_height, im_width = im.shape[:2]

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    im_blurred = cv2.GaussianBlur(im_gray, (5, 5), 0)

    im_edged = cv2.Canny(im_blurred, 30, 150)

    (im_contoured, contours, _) = cv2.findContours(im_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted([(contour, cv2.boundingRect(contour)[0]) for contour in contours], key =lambda x: x[1])

    rectangles = [cv2.boundingRect(contour[0]) for contour in contours]

    leeway = 50
    rectangles, bad_rectangles = valid_rectangles(rectangles, leeway)

    while bad_rectangles > 0:
        rectangles, bad_rectangles = valid_rectangles(rectangles, leeway)

    classification = ''
    for rectangle in rectangles:
        x, y, w, h = rectangle

        if 7<= w <= im_width / 4 and 10 <= h <= im_height / 4:
            roi = im_gray[y: y + h, x: x + w]
            im_thresh = roi.copy()
            threshold = mahotas.thresholding.otsu(roi)
            im_thresh[im_thresh > threshold] = 255
            im_thresh = cv2.bitwise_not(im_thresh)

            im_thresh = dataset.center_extent(im_thresh, (28, 28))
            # cv2.imshow('idk', im_thresh)
            # cv2.waitKey(0)

            # CNN
            # im_thresh = cv2.copyMakeBorder(im_thresh , 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0, 0, 0)) #28 x 28 image
            network_input = im_thresh.reshape(1, 1, 28, 28)
            network_output = f_output(network_input).reshape([14]).tolist()
            print network_output
            confidence = max(network_output)
            prediction = network_output.index(confidence)
            character = possible_values[prediction]

            print 'Predicted: {} with {} confidence.'.format(character, confidence)

            print '-' * 100
            color = (128, 128, 128)
            if confidence >= .98:
                color = (255, 0, 0)
            elif .5 < confidence < .98:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            # SVM
            # hist = hog.describe(im_thresh)
            # output_vector = svm.predict_proba(hist).reshape([14]).tolist()
            # confidence = max(output_vector)
            # prediction = output_vector.index(confidence)
            # character = possible_values[prediction]
            #
            # color = (128, 128, 128)
            # if confidence >= .98:
            #     color = (255, 0, 0)
            # elif .5 < confidence < .98:
            #     color = (0, 255, 0)
            # else:
            #     color = (0, 0, 255)

            cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)

            cv2.putText(im, str(character), (x - 10 + w / 2, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            classification += str(character) + ' '

    classification = classification[:-1]
    print '-> Classification: ', classification

    if __name__ == '__main__':
        cv2.imshow('image', im)
        cv2.waitKey(0)

    # Saves processed image
    image_path, extension = os.path.splitext(problem.image.path)
    processed_image_path = image_path + '_processed' + extension
    cv2.imwrite(processed_image_path, im)

    # Updated the problem object with the processed photo
    opened = open(processed_image_path, 'rb')
    processed_image = File(opened)
    problem.processed_image.save(processed_image_path, processed_image)
    problem.save()
    os.remove(processed_image_path)

    classification = classification.strip('\r\n')

    prediction_solution = ''

    try:
        prediction_solution = eval(classification)
    except SyntaxError:
        prediction_solution = 'Error'
    finally:
        prediction_solution = str(prediction_solution)

    return classification, prediction_solution

if __name__ == '__main__':
    problems = Problem.objects.all().order_by('-id')[:10]
    for problem in problems:
        classify_problem(problem.id)
        cv2.destroyAllWindows()
        print '-' * 100

    # # problems = Problem.objects.all()
    # # contour_detection(problems[0].id)
    # classify_problem(206) #minus
    # classify_problem(207) #plus training 1
    # classify_problem(203) #plus training 2
    #
    # classify_problem(182)
    # classify_problem(177)
    # classify_problem(190)
    # classify_problem(226)
    # classify_problem(255)

    # plus_training_samples = [289, 290, 291, 292, 293, 294, 295]