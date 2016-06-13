"""Attempt to show "stroke"-like visualizations by combining only real
test cases as a basis.
"""
import logging
from argparse import ArgumentParser

from theano import tensor

from blocks.bricks.conv import Convolutional
from blocks.bricks import Linear
from blocks.bricks import Rectifier
from blocks.bricks import Activation
from blocks.bricks import Softmax
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter
from blocks.filter import get_brick
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.serialization import load_parameters
from blocks.utils import shared_floatx
from collections import OrderedDict
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from intent.lenet import LeNet
from intent.maxact import MaximumActivationSearch
from intent.filmstrip import Filmstrip
from intent.rf import center_location
from intent.rf import layerarray_fieldmap
from theano import gradient
from theano import tensor
from prior import create_fair_basis
from prior import make_shifted_basis
import theano
import numpy
import numbers

# For testing
from blocks.roles import OUTPUT

def main(save_to):
    batch_size = 365
    feature_maps = [6, 16]
    mlp_hiddens = [120, 84]
    conv_sizes = [5, 5]
    pool_sizes = [2, 2]
    image_size = (28, 28)
    output_size = 10

    # The above are from LeCun's paper. The blocks example had:
    #    feature_maps = [20, 50]
    #    mlp_hiddens = [500]

    # Use ReLUs everywhere and softmax for the final prediction
    conv_activations = [Rectifier() for _ in feature_maps]
    mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()]
    convnet = LeNet(conv_activations, 1, image_size,
                    filter_sizes=zip(conv_sizes, conv_sizes),
                    feature_maps=feature_maps,
                    pooling_sizes=zip(pool_sizes, pool_sizes),
                    top_mlp_activations=mlp_activations,
                    top_mlp_dims=mlp_hiddens + [output_size],
                    border_mode='valid',
                    weights_init=Uniform(width=.2),
                    biases_init=Constant(0))
    # We push initialization config to set different initialization schemes
    # for convolutional layers.
    convnet.push_initialization_config()
    convnet.layers[0].weights_init = Uniform(width=.2)
    convnet.layers[1].weights_init = Uniform(width=.09)
    convnet.top_mlp.linear_transformations[0].weights_init = Uniform(width=.08)
    convnet.top_mlp.linear_transformations[1].weights_init = Uniform(width=.11)
    convnet.initialize()
    logging.info("Input dim: {} {} {}".format(
        *convnet.children[0].get_dim('input_')))
    for i, layer in enumerate(convnet.layers):
        if isinstance(layer, Activation):
            logging.info("Layer {} ({})".format(
                i, layer.__class__.__name__))
        else:
            logging.info("Layer {} ({}) dim: {} {} {}".format(
                i, layer.__class__.__name__, *layer.get_dim('output')))

    random_init = (numpy.random.rand(100, 1, 28, 28) * 128).astype('float32')
    layers = [l for l in convnet.layers if isinstance(l, Convolutional)]
    mnist_test = MNIST(("test",), sources=['features', 'targets'])
    basis_init = create_fair_basis(mnist_test, 10, 50)
    basis_set = make_shifted_basis(basis_init, convnet, layers)

    for layer, basis in zip(layers, basis_set):
        # basis is 5d:
        # (probed_units, base_cases, 1-c, 28-y, 28-x)
        b = shared_floatx(basis)
        # coefficients is 2d:
        # (probed_units, base_cases)
        coefficients = shared_floatx(
                numpy.ones(basis.shape[0:2],
                    dtype=theano.config.floatX))
        # prod is 5d: (probed_units, base_cases, 1-c, 28-y, 28-x)
        prod = tensor.shape_padright(coefficients, 3) * b
        # x is 4d: (probed_units, 1-c, 28-y, 28-x)
        ux = prod.sum(axis=1)
        x = tensor.clip(ux /
                tensor.shape_padright(ux.flatten(ndim=2).max(axis=1), 3),
                0, 1)

        # Normalize input and apply the convnet
        probs = convnet.apply(x)
        cg = ComputationGraph([probs])
        outs = VariableFilter(
                roles=[OUTPUT], bricks=[layer])(cg.variables)

        # Create an interior activation model
        model = Model([probs] + outs)

        # Load it with trained parameters
        params = load_parameters(open(save_to, 'rb'))
        model.set_parameter_values(params)

        learning_rate = shared_floatx(0.03, 'learning_rate')
        # We will try to do all units at once.
        # unit = shared_floatx(0, 'unit', dtype='int64')
        # But we are only doing one layer at once.
        output = outs[0]
        dims = layer.get_dims(['output'])[0]
        if isinstance(dims, numbers.Integral):
            # FC case: output is 2d: (probed_units, units)
            dims = (dims, )
            unitrange = tensor.arange(dims[0])
            costvec = -tensor.log(
                    tensor.nnet.softmax(output)[unitrange, unitrage].
                    flatten())
        else:
            # Conv case: output is 4d: (probed_units, units, y, x)
            unitrange = tensor.arange(dims[0])
            print('dims is', dims)
            costvec = -tensor.log(tensor.nnet.softmax(output[
                unitrange, unitrange, dims[1] // 2, dims[2] // 2]).
                flatten())
        cost = costvec.sum()
        # grad is dims (probed_units, basis_size)
        grad = gradient.grad(cost, coefficients)
        stepc = coefficients # - learning_rate * grad
        newc = stepc / tensor.shape_padright(stepc.mean(axis=1))
        fn = theano.function([], [cost, x], updates=[(coefficients, newc)])
        filmstrip = Filmstrip(
            random_init.shape[-2:], (dims[0], 1),
            background='red')
        layer = get_brick(output)
        learning_rate.set_value(0.1)
        for index in range(20000):
            c, result = fn()
            if index % 1000 == 0:
                learning_rate.set_value(numpy.cast[theano.config.floatX](
                    learning_rate.get_value() * 0.8))
                print('cost', c)
                for u in range(dims[0]):
                    filmstrip.set_image((0, u), result[u,:,:,:])
                    filmstrip.save(layer.name + '_stroke.jpg')
            for u in range(dims[0]):
                filmstrip.set_image((0, u), result[u,:,:,:])
            filmstrip.save(layer.name + '_stroke.jpg')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("Gradient descent vis for the MNIST dataset.")
    parser.add_argument("save_to", default="mnist.tar", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    args = parser.parse_args()
    main(**vars(args))
