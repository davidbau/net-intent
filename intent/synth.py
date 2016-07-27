"""Convolutional network example.

Synthesize gradient descent images by

```
python synth.py
```

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
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from intent.maxact import MaximumActivationSearch
from intent.filmstrip import Filmstrip
from intent.rf import make_mask
from intent.rf import layerarray_fieldmap
from prior import create_fair_basis
from theano import gradient
from theano import tensor
from intent.noisy import create_noisy_lenet_5
import theano
import numpy
import numbers

# For testing
from blocks.roles import OUTPUT

def add_noise(params):
    for p in params:
        if p.endswith('.N'):
            params[p] = numpy.random.normal(params[p].shape).astype(
                    params[p].dtype)

def zero_noise(params):
    for p in params:
        if p.endswith('.N'):
            params[p] = numpy.zeros(params[p].shape, dtype=params[p].dtype)

def main(save_to):
    batch_size = 500
    image_size = (28, 28)
    output_size = 10

    # The above are from LeCun's paper. The blocks example had:
    #    feature_maps = [20, 50]
    #    mlp_hiddens = [500]

    # Use ReLUs everywhere and softmax for the final prediction
    convnet = create_noisy_lenet_5(batch_size)

    mnist_test = MNIST(("test",), sources=['features', 'targets'])
    basis_init = create_fair_basis(mnist_test, 10, 2)

    # b = shared_floatx(basis)
    # random_init = numpy.rand.random(100, 1000)
    # r = shared_floatx(random_init)
    # rn = r / r.norm(axis=1)
    # x = tensor.dot(rn, tensor.shape_padright(b))
    x = shared_floatx(basis_init)

    # Normalize input and apply the convnet
    probs = convnet.apply(x)
    cg = ComputationGraph([probs])
    outs = VariableFilter(
            roles=[OUTPUT], bricks=[Convolutional, Linear])(cg.variables)

    # Create an interior activation model
    model = Model([probs] + outs)

    # Load it with trained parameters
    params = load_parameters(open(save_to, 'rb'))
    # Zero any noise
    zero_noise(params)
    model.set_parameter_values(params)

    learning_rate = shared_floatx(0.01, 'learning_rate')
    unit = shared_floatx(0, 'unit', dtype='int64')
    negate = False
    suffix = '_negsynth.jpg' if negate else '_synth.jpg'
    for output in outs:
        layer = get_brick(output)
        # For now, skip masks -for some reason they are always NaN
        iterations = 10000
        if layer.name == 'mask':
            continue
            # iterations = 20000
        layername = layer.parents[0].name + '-' + layer.name
        # if layername != 'noisylinear_2-linear':
        #     continue
        dims = layer.get_dims(['output'])[0]
        if negate:
            measure = -output
        else:
            measure = output
        measure = measure[(slice(0, basis_init.shape[0]), ) +
                (slice(None),) * (measure.ndim - 1)]
        if isinstance(dims, numbers.Integral):
            dims = (dims, )
            costvec = -tensor.log(tensor.nnet.softmax(
                measure)[:,unit].flatten())
        else:
            flatout = measure.flatten(ndim=3)
            maxout = flatout.max(axis=2)
            costvec = -tensor.log(tensor.nnet.softmax(
                maxout)[:,unit].flatten())
        # Add a regularization to favor gray images.
        print(layer, layer.get_hierarchical_name(output), output, dims)
        # cost = costvec.sum() + (x - 0.5).norm(2) * (
        #         10.0 / basis_init.shape[0])
        cost = costvec.sum()
        grad = gradient.grad(cost, x)
        stepx = x - learning_rate * grad
        normx = stepx / tensor.shape_padright(
                stepx.flatten(ndim=2).max(axis=1), n_ones=3)
        newx = tensor.clip(normx, 0, 1)
        newx = newx[(slice(0, basis_init.shape[0]), ) +
                (slice(None),) * (newx.ndim - 1)]
        fn = theano.function([], [cost], updates=[(x, newx)])
        filmstrip = Filmstrip(
            basis_init.shape[-2:], (dims[0], basis_init.shape[0]),
            background='red')
        for u in range(dims[0]):
            unit.set_value(u)
            x.set_value(basis_init)
            print('layer', layername, 'unit', u)
            for index in range(iterations):
                add_noise(params)
                c = fn()[0]
                if index % 1000 == 0:
                    print('cost', c)
                    result = x.get_value()
                    for i2 in range(basis_init.shape[0]):
                        filmstrip.set_image((u, i2), result[i2,:,:,:])
                    filmstrip.save(layername + suffix)
            result = x.get_value()
            for index in range(basis_init.shape[0]):
                filmstrip.set_image((u, index), result[index,:,:,:])
            filmstrip.save(layer.name + suffix)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("Gradient descent vis for the MNIST dataset.")
    parser.add_argument("save_to", default="noisy-mnist.tar", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    args = parser.parse_args()
    main(**vars(args))
