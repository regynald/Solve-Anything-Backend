from __future__ import division

import os
import sys
import time
import django
import cv2

import cPickle as pickle
import numpy as np

from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.datasets import fetch_mldata

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ['DJANGO_SETTINGS_MODULE'] = 'SolveAnythingBackend.settings'
django.setup()

from SolveAnythingBackend import settings
from SolveAnything.Classifier.SVM.hog import HOG
from SolveAnything.Classifier import dataset

MODELS_DIR = settings.BASE_DIR + '/SolveAnything/Classifier/models/SVM/'
DATA_DIR = settings.BASE_DIR + '/SolveAnything/Classifier/data/'

def load_mnist_subset():
    filename = 'digits.csv'
    data = np.genfromtxt(settings.BASE_DIR+'/SolveAnything/Classifier/data/'+filename, delimiter = ",", dtype = "uint8")
    target = data[:, 0]
    data = data[:, 1:].reshape(-1, 28, 28)
    return data, target

def load_pure_mnist():
    mnist = fetch_mldata('mnist-original', data_home=DATA_DIR)
    data = mnist.data
    target = mnist.target
    data = data.reshape((-1, 28, 28))
    target = target.astype(np.uint8)
    return data, target

def load_dataset(randomize = False, overfit = False):
    mnist = fetch_mldata('mnist-original', data_home=DATA_DIR)
    data = mnist.data
    target = mnist.target
    data = data.reshape((-1, 28, 28))
    target = target.astype(np.uint8)

    operator_train = pickle.load(open(DATA_DIR + 'four_operators.pickle', 'rb'))
    operator_data = np.array([x[0].reshape((28, 28)) for x in operator_train])
    operator_target = np.array([y[1] for y in operator_train])

    # # # Overfitting meme
    if overfit == True:
        temp_data = operator_data
        temp_target = operator_target
        while operator_data.shape[0] < 20000:
            operator_data = np.concatenate((operator_data, temp_data))
            operator_target = np.concatenate((operator_target, temp_target))
    # # # Overfitting meme

    data = np.concatenate((data, operator_data))
    target = np.concatenate((target, operator_target))

    if randomize:
        print 'shuffling data'
        data, target = shuffle(data, target, random_state=0)

    target = target.astype(np.uint8)

    return data, target

def train_classifier(filename, randomize = False, overfit = False, verbose = False):
    data, target = load_dataset(randomize=randomize, overfit=overfit)
    hist_data = []

    hog = HOG(orientations=18, pixelsPerCell=(10, 10), cellsPerBlock=(1, 1), normalize=True)

    if os.path.exists(MODELS_DIR + filename):
        print 'loading model'
        model = joblib.load(MODELS_DIR + filename)
    else:
        print 'training model'
        i = 0
        for image in data:
            image = dataset.center_extent(image, (20, 20))
            hist = hog.describe(image)
            hist_data.append(hist)
            i += 1

        # model = LinearSVC(random_state=42)
        model = svm.SVC(probability=True, kernel='sigmoid')
        # model = svm.NuSVC(probability=True)
        start = time.time()
        model.fit(hist_data, target)

        seconds = time.time() - start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        print "Took %d:%02d:%02d to train model." % (h, m, s)

        joblib.dump(model, MODELS_DIR + filename)

    correct = 0
    total = 0

    for i in range(data.shape[0]):
        image = data[i]
        input_image = image
        input_image = dataset.center_extent(input_image, (20, 20))
        hist = hog.describe(input_image)
        pred = model.predict_proba(hist).reshape([14]).tolist()
        max_value = max(pred)
        max_index = pred.index(max_value)

        if verbose:
            print '-' * 50
            print 'Predicted: ' + str(max_index) + ' with ' + str(max_value) + ' confidence'
            print 'Actual: ' + str(target[i])

            cv2.imshow('input', image)
            cv2.waitKey(0)

        if max_index == target[i]:
            correct += 1
        total += 1

    print str(correct) + ' out of ' + str(total)
    print 'accuracy: ' + str(correct / total)

    return model

train_classifier('svm-sigmoid-SVC-solve-anything-model-01-21-2016', randomize=True, verbose=False)