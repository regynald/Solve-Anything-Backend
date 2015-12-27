import os
import sys
import django

from sklearn.externals import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ['DJANGO_SETTINGS_MODULE'] = 'SolveAnythingBackend.settings'
django.setup()

from SolveAnythingBackend import settings
from SolveAnything.Classifier.SVM.hog import HOG

MODELS_DIR = settings.BASE_DIR + '/SolveAnything/Classifier/models/SVM/'


def loadSupportVectorMachineFromFile(filename='svm-solve-anything-model-01-21-2016'):
    model = joblib.load(MODELS_DIR + filename)
    hog = HOG(orientations = 18, pixelsPerCell = (10, 10), cellsPerBlock = (1, 1), normalize = True)
    return model, hog

