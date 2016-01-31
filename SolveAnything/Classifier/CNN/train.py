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
from urllib import urlretrieve

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
    data = data[:, 1:].reshape(-1, 1, 28, 28)
    return data, target


def load_pure_mnist():
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, DATA_DIR + filename)

    def load_mnist_images(filename):
        if not os.path.exists(DATA_DIR + filename):
            download(filename)
        with gzip.open(DATA_DIR + filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(DATA_DIR + filename):
            download(filename)
        with gzip.open(DATA_DIR + filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    x_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    x_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    x_data = np.concatenate((x_train, x_test))
    y_target = np.concatenate((y_train, y_test))
    return x_data, y_target


def load_dataset(randomize=False):
    mnist_data, mnist_target = load_pure_mnist()

    operator_train = pickle.load(open(DATA_DIR + 'four_operators.pickle', 'rb'))
    operator_data = np.array([x[0] for x in operator_train])
    operator_target = np.array([y[1] for y in operator_train])

    operator_data = operator_data.reshape(-1, 1, 28, 28)
    operator_data = operator_data / np.float32(256)
    operator_target = operator_target.astype(np.uint8)

    data = np.concatenate((mnist_data, operator_data))
    target = np.concatenate((mnist_target, operator_target))

    if randomize:
        print 'shuffling data'
        data, target = shuffle(data, target, random_state=0)

    return data, target


def train_network(filename, randomize=False, verbose=False):
    data, target = load_dataset(randomize=randomize)

    net1 = createNeuralNet(type='deep')

    if os.path.exists(MODELS_DIR + filename):
        print 'loading model'
        net1.load_params_from(MODELS_DIR + filename)
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
        network_input = instance[None, :, :]
        network_pred = f_output(network_input).reshape([14]).tolist()
        max_value = max(network_pred)
        max_index = network_pred.index(max_value)

        if verbose:
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

net11 = train_network(filename='deep-cnn-solve-anything-model-01-30-16.pkl')

# for i in range(100):
    # train_network(filename='test-network' + str(i) + '.pkl')