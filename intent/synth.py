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
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from intent.lenet import LeNet
from intent.maxact import MaximumActivationSearch
from intent.filmstrip import Filmstrip
from intent.rf import make_mask
from intent.rf import layerarray_fieldmap
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

    mnist_test = MNIST(("test",), sources=['features'])
    mnist_test_stream = DataStream.default_stream(
        mnist_test,
        iteration_scheme=SequentialScheme(
            mnist_test.num_examples, 100))
    random_init = (next(mnist_test_stream.get_epoch_iterator())[0])

    # b = shared_floatx(basis)
    # random_init = numpy.rand.random(100, 1000)
    # r = shared_floatx(random_init)
    # rn = r / r.norm(axis=1)
    # x = tensor.dot(rn, tensor.shape_padright(b))
    x = shared_floatx(random_init)

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

    learning_rate = shared_floatx(0.001, 'learning_rate')
    unit = shared_floatx(0, 'unit', dtype='int64')
    for output in outs:
        dims = get_brick(output).get_dims(['output'])[0]
        if isinstance(dims, numbers.Integral):
            dims = (dims, )
            costvec = -tensor.log(tensor.nnet.softmax(output)[:,unit].flatten())
        else:
            flatout = output.flatten(ndim=3)
            maxout = flatout.max(axis=2)
            costvec = -tensor.log(tensor.nnet.softmax(maxout)[:,unit].flatten())
        cost = costvec.sum()
        grad = gradient.grad(cost, x)
        stepx = x - learning_rate * grad
        normx = stepx / tensor.shape_padright(
                stepx.flatten(ndim=2).max(axis=1), n_ones=3)
        newx = tensor.clip(normx, 0, 1)
        fn = theano.function([], [cost], updates=[(x, newx)])
        filmstrip = Filmstrip(
            random_init.shape[-2:], (dims[0], random_init.shape[0]),
            background='red')
        layer = get_brick(output)
        for u in range(dims[0]):
            unit.set_value(u)
            x.set_value(random_init)
            print('layer', layer.name, 'unit', u)
            for index in range(100000):
                c = fn()[0]
                if index % 10000 == 0:
                    print('cost', c)
                    result = x.get_value()
                    for i2 in range(100):
                        filmstrip.set_image((i2, u), result[i2,:,:,:] * 255)
                    filmstrip.save(layer.name + '_synth.jpg')
            result = x.get_value()
            for index in range(100):
                filmstrip.set_image((index, u), result[index,:,:,:] * 255)
            filmstrip.save(layer.name + '_synth.jpg')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("Gradient descent vis for the MNIST dataset.")
    parser.add_argument("save_to", default="mnist.tar", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    args = parser.parse_args()
    main(**vars(args))
