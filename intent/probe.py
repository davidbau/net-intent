"""Convolutional network example.

Synthesize gradient descent images by

```
python probe.py
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
from blocks.bricks.cost import MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter
from blocks.filter import get_brick
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring.evaluators import DatasetEvaluator
from blocks.serialization import load_parameters
from blocks.utils import shared_floatx
from blocks.utils import dict_subset
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from intent.lenet import LeNet
from intent.maxact import MaximumActivationSearch
from intent.filmstrip import Filmstrip
from intent.rf import make_mask
from intent.rf import layerarray_fieldmap
from prior import create_fair_basis
from theano import gradient
from theano import tensor
from intent.ablation import ConfusionMatrix
from intent.ablation import Sum
from intent.ablation import ablate_inputs
from collections import OrderedDict
import theano
import numpy
import numbers
import pickle

# For testing
from blocks.roles import OUTPUT

def extract_sample(activations, data_stream, n=2000):
    cg = ComputationGraph(activations)
    input_names = [v.name for v in cg.inputs]
    fn = theano.function(cg.inputs, [activations])
    result = None
    for batch in data_stream.get_epoch_iterator(as_dict=True):
        values = fn(**dict_subset(batch, input_names))
        if result is None:
            result = values[0]
        else:
            result = numpy.concatenate((result, values[0]))
        if result.shape[0] >= n:
            result = result[(slice(0, n), ) +
                    (slice(None),) * (len(result.shape) - 1)]
            return result

def main(save_to, hist_file):
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

    mnist_test = MNIST(("test",), sources=['features', 'targets'])

    x = tensor.tensor4('features')
    y = tensor.lmatrix('targets')

    # Normalize input and apply the convnet
    probs = convnet.apply(x)
    error_rate = (MisclassificationRate().apply(y.flatten(), probs)
                  .copy(name='error_rate'))
    confusion = (ConfusionMatrix().apply(y.flatten(), probs)
                  .copy(name='confusion'))
    confusion.tag.aggregation_scheme = Sum(confusion)

    model = Model([error_rate, confusion])

    # Load it with trained parameters
    params = load_parameters(open(save_to, 'rb'))
    model.set_parameter_values(params)

    def full_brick_name(brick):
        return '/'.join([''] + [b.name for b in brick.get_unique_path()])

    # Find layer outputs to probe
    outs = OrderedDict((full_brick_name(get_brick(out)), out)
            for out in VariableFilter(
                roles=[OUTPUT], bricks=[Convolutional, Linear])(
                    model.variables))

    # Load histogram information
    with open(hist_file, 'rb') as handle:
        histograms = pickle.load(handle)

    # Corpora
    mnist_train = MNIST(("train",))
    mnist_train_stream = DataStream.default_stream(
        mnist_train, iteration_scheme=ShuffledScheme(
            mnist_train.num_examples, batch_size))

    mnist_test = MNIST(("test",))
    mnist_test_stream = DataStream.default_stream(
        mnist_test,
        iteration_scheme=ShuffledScheme(
            mnist_test.num_examples, batch_size))

    # Probe the given layer
    target_layer = '/lenet/mlp/linear_0'
    next_layer_param = '/lenet/mlp/linear_1.W'
    sample = extract_sample(outs[target_layer], mnist_test_stream)
    print('sample shape', sample.shape)

    # Figure neurons to ablate
    hist = histograms[('linear_1', 'b')]
    targets = [i for i in range(hist.shape[1])
            if hist[2, i] * hist[7, i] < 0]
    print('ablating', len(targets), ':', targets)

    # Now adjust the next layer weights based on the probe
    param = model.get_parameter_dict()[next_layer_param]
    print('param shape', param.get_value().shape)

    new_weights = ablate_inputs(
        targets,
        sample,
        param.get_value(),
        compensate=False)
    param.set_value(new_weights)

    # Evaluation pass
    evaluator = DatasetEvaluator([error_rate, confusion])
    print(evaluator.evaluate(mnist_test_stream))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("Gradient descent vis for the MNIST dataset.")
    parser.add_argument("save_to", default="mnist.tar", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    parser.add_argument("--hist-file", default="histograms120.pkl", nargs="?",
                        help="Histograms for reading out neuron intent")
    args = parser.parse_args()
    main(**vars(args))
