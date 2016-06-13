"""Convolutional network example.

Run the training for 20 epochs with
```
python run.py --num-epochs 20
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
from collections import OrderedDict
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from intent.lenet import LeNet
from intent.maxact import MaximumActivationSearch
from intent.filmstrip import Filmstrip
from intent.rf import make_mask
from intent.rf import layerarray_fieldmap
from prior import create_fair_basis
from theano import gradient
from theano import tensor
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
    layers = convnet.layers

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

    mnist_test = MNIST(("test",), sources=['features', 'targets'])
    basis = create_fair_basis(mnist_test, 10, 10)

#    state = mnist_test.open()
#
#    basis = numpy.zeros((100, 1, 28, 28), dtype=theano.config.floatX)
#    counters = [0] * 10
#    index = 0
#    while min(counters) < 10:
#        feature, target = mnist_test.get_data(state=state, request=[index])
#        target = target[0, 0]
#        feature = feature / 256
#        if counters[target] < 10:
#            basis[target + counters[target] * 10, :, :, :] = feature[0, :, :, :]
#            counters[target] += 1
#        index += 1
#    mnist_test.close(state=state)

    
    # b = shared_floatx(basis)
    # random_init = numpy.rand.random(100, 1000)
    # r = shared_floatx(random_init)
    # rn = r / r.norm(axis=1)
    # x = tensor.dot(rn, tensor.shape_padright(b))
    x = tensor.tensor4('features')

    # Normalize input and apply the convnet
    probs = convnet.apply(x)
    cg = ComputationGraph([probs])
    outs = VariableFilter(
            roles=[OUTPUT], bricks=[Convolutional, Linear])(cg.variables)

    # Create an interior activation model
    model = Model([probs] + outs)

    # Load it with trained parameters
    params = load_parameters(open(save_to, 'rb'))
    model.set_parameter_values(params)
    fn = theano.function([x], outs)
    results = fn(basis)
    for snapshots, output in zip(results, outs):
        layer = get_brick(output)
        filmstrip = Filmstrip(
            basis.shape[-2:], (snapshots.shape[1], snapshots.shape[0]),
            background='purple')

        if layer in layers:
            fieldmap = layerarray_fieldmap(layers[0:layers.index(layer) + 1])
            for unit in range(snapshots.shape[1]):
                for index in range(snapshots.shape[0]):
                    mask = make_mask(basis.shape[-2:], fieldmap, numpy.clip(
                        snapshots[index, unit, :, :], 0, numpy.inf))
                    filmstrip.set_image((index, unit),
                        basis[index, :, :, :], mask)
            filmstrip.save(layer.name + '_show.jpg')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("Gradient descent vis for the MNIST dataset.")
    parser.add_argument("save_to", default="mnist.tar", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    args = parser.parse_args()
    main(**vars(args))
