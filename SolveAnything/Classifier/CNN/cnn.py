import os
import sys
import django
import lasagne
import theano

from lasagne import layers
from nolearn.lasagne import NeuralNet
from lasagne.updates import nesterov_momentum

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ['DJANGO_SETTINGS_MODULE'] = 'SolveAnythingBackend.settings'
django.setup()

from SolveAnythingBackend import settings

MODELS_DIR = settings.BASE_DIR + '/SolveAnything/Classifier/models/CNN/'

def shallowNetwork():
    layers1 = [
        ('input', layers.InputLayer),
        ('conv2d1', layers.Conv2DLayer),
        ('maxpool1', layers.MaxPool2DLayer),
        ('conv2d2', layers.Conv2DLayer),
        ('maxpool2', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('dense', layers.DenseLayer),
        ('dropout2', layers.DropoutLayer),
        ('output', layers.DenseLayer),
    ]

    net1 = NeuralNet(
        layers=layers1,
        # input layer
        input_shape=(None, 1, 28, 28),
        # layer conv2d1
        conv2d1_num_filters=32,
        conv2d1_filter_size=(5, 5),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=lasagne.init.GlorotUniform(),
        # layer maxpool1
        maxpool1_pool_size=(2, 2),
        # layer conv2d2
        conv2d2_num_filters=32,
        conv2d2_filter_size=(5, 5),
        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        # layer maxpool2
        maxpool2_pool_size=(2, 2),
        # dropout1
        dropout1_p=0.5,
        # dense
        dense_num_units=256,
        dense_nonlinearity=lasagne.nonlinearities.rectify,
        # dropout2
        dropout2_p=0.5,
        # output
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=14,
        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
        max_epochs=100,
        verbose=1,
    )
    return net1

def deepNetwork():
    layers2 = [
        ('input', layers.InputLayer),
        ('conv2d1', layers.Conv2DLayer),
        ('conv2d2', layers.Conv2DLayer),
        ('conv2d3', layers.Conv2DLayer),
        ('conv2d4', layers.Conv2DLayer),
        ('conv2d5', layers.Conv2DLayer),
        ('conv2d6', layers.Conv2DLayer),
        ('conv2d7', layers.Conv2DLayer),
        ('maxpool1', layers.MaxPool2DLayer),
        ('conv2d8', layers.Conv2DLayer),
        ('conv2d9', layers.Conv2DLayer),
        ('conv2d10', layers.Conv2DLayer),
        ('maxpool2', layers.MaxPool2DLayer),
        ('dense1', layers.DenseLayer),
        ('dropout1', layers.DropoutLayer),
        ('dense2', layers.DenseLayer),
        ('output', layers.DenseLayer),
    ]

    net2 = NeuralNet(
        layers=layers2,
        # input layer
        input_shape=(None, 1, 28, 28),
        # layer conv2d1
        conv2d1_num_filters=32,
        conv2d1_filter_size=(3, 3),
        conv2d1_pad = 1,
        # layer conv2d2
        conv2d2_num_filters=32,
        conv2d2_filter_size=(3, 3),
        conv2d2_pad = 1,
        # layer conv2d3
        conv2d3_num_filters=32,
        conv2d3_filter_size=(3, 3),
        conv2d3_pad = 1,
        # layer conv2d4
        conv2d4_num_filters=32,
        conv2d4_filter_size=(3, 3),
        conv2d4_pad = 1,
        # layer conv2d5
        conv2d5_num_filters=32,
        conv2d5_filter_size=(3, 3),
        conv2d5_pad = 1,
        # layer conv2d6
        conv2d6_num_filters=32,
        conv2d6_filter_size=(3, 3),
        conv2d6_pad = 1,
        # layer conv2d7
        conv2d7_num_filters=32,
        conv2d7_filter_size=(3, 3),
        conv2d7_pad = 1,
        # layer maxpool1
        maxpool1_pool_size=(2, 2),
        # layer conv2d8
        conv2d8_num_filters=32,
        conv2d8_filter_size=(3, 3),
        conv2d8_pad = 1,
        # layer conv2d9
        conv2d9_num_filters=32,
        conv2d9_filter_size=(3, 3),
        conv2d9_pad = 1,
        # layer conv2d10
        conv2d10_num_filters=32,
        conv2d10_filter_size=(3, 3),
        conv2d10_pad = 1,
        # layer maxpool2
        maxpool2_pool_size=(2, 2),
        # dense1
        dense1_num_units=64,
        # dropout1
        dropout1_p=0.5,
        # dense2
        dense2_num_units=64,
        # output
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=14,
        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
        max_epochs=30,
        verbose=2,
    )
    return net2

def createNeuralNet():
    return shallowNetwork()

def loadNeuralNetworkFromFile(filename='shallow-cnn-solve-anything-model2-01-27-16.pkl'):
    net1 = createNeuralNet()
    if os.path.exists(MODELS_DIR + filename):
        nn = net1.load_params_from(MODELS_DIR + filename)
        output_layer = layers.get_output(net1.layers_['output'], deterministic=True)
        input_var = net1.layers_['input'].input_var
        f_output = theano.function([input_var], output_layer)
        return nn, f_output
    else:
        return None, None
