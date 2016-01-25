from __future__ import division

import os
import sys
import django
import gzip
import theano
import cv2

import cPickle as pickle
import numpy as np

from lasagne import layers
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ['DJANGO_SETTINGS_MODULE'] = 'SolveAnythingBackend.settings'
django.setup()

from SolveAnythingBackend import settings
from SolveAnything.Classifier.CNN.cnn import createNeuralNet

MODELS_DIR = settings.BASE_DIR + '/SolveAnything/Classifier/models/CNN/'
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


def train_network(filename, randomize=False, overfit=False, verbose=True):
    data, target = load_dataset(randomize=randomize, overfit=overfit)
    net1 = createNeuralNet()

    if os.path.exists(MODELS_DIR + filename):
        print 'loading model'
        nn = net1.load_params_from(MODELS_DIR + filename)
    else:
        print 'training model'
        nn = net1.fit(data, target)
        nn.save_params_to(MODELS_DIR + filename)

    output_layer = layers.get_output(net1.layers_['output'], deterministic=True)
    input_var = net1.layers_['input'].input_var

    f_output = theano.function([input_var], output_layer)

    correct = 0
    total = 0
    for i in range(data.shape[0]):
        instance = data[i]
        input = instance[None, :, :]
        pred = f_output(input).reshape([14]).tolist()
        max_value = max(pred)
        max_index = pred.index(max_value)

        print 'Predicted: ' + str(max_index) + ' with ' + str(max_value) + ' confidence'
        print 'Actual: ' + str(target[i])

        instance_image = instance.reshape([28, 28])
        cv2.imshow('instance', instance_image)
        cv2.waitKey(0)
        print '-' * 50

        if max_index == target[i]:
            correct += 1
        total += 1

    print str(correct) + ' out of ' + str(total)
    print 'accuracy: ' + str(correct / total)
    print '-' * 50


    return net1

net11 = train_network(filename='deep-cnn-solve-anything-model-01-19-2016.pkl')
