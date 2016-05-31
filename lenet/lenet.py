"""Convolutional network example.

Run the training for 50 epochs with
```
python __init__.py --num-epochs 50
```
It is going to reach around 0.8% error rate on the test set.

"""
import logging
import numpy as np
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict

from theano import tensor
from theano import gradient

from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import (MLP, Rectifier, Initializable, FeedforwardSequence,
                           Softmax, Activation, Linear, application)
from blocks.bricks.base import Brick
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence,
                                Flattener, MaxPooling)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.roles import add_role, PersistentRole, WEIGHT
from blocks.utils import shared_floatx_zeros
from fuel.datasets import MNIST
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from picklable_itertools.extras import equizip
from toolz.itertoolz import interleave

class ComponentwiseCrossEntropy(Brick):
    @application(outputs=["components"])
    def apply(self, y, y_hat):
        # outputs a vector with cross entropy for each row
        total = tensor.nnet.crossentropy_categorical_1hot(y_hat, y)
        # now place the cross entropy in the column for that class
        splitout = (
            total.dimshuffle(0, 'x') *
            tensor.extra_ops.to_one_hot(y, y_hat.shape[1]))
        # now average the cross entropy componentwise
        components = splitout.mean(axis=0)
        return components

class AttributionStatisticsRole(PersistentRole):
    pass

# role for attribution historgram
ATTRIBUTION_STATISTICS = AttributionStatisticsRole()



def _create_attribution_histogram_for(param, components_size):
    buf = shared_floatx_zeros((components_size,) + param.get_value().shape)
    buf.tag.for_parameter = param
    add_role(buf, ATTRIBUTION_STATISTICS)
    return buf

def _create_attribution_updates(attribution, jacobian):
    return (attribution, attribution + jacobian)

class AttributedGradientDescent(GradientDescent):
    def __init__(self, components=None, components_size=None,
                 jacobians=None, **kwargs):
        super(AttributedGradientDescent, self).__init__(**kwargs)
        self.components = components
        self.components_size = components_size
        self.jacobians = jacobians
        if not self.jacobians:
            self.jacobians = self._compute_jacobians()
        self.attributions = OrderedDict(
            [(param, _create_attribution_histogram_for(param, components_size))
                for param in self.parameters])
        self.attribution_updates = OrderedDict(
            [_create_attribution_updates(self.attributions[param],
                self.jacobians[param]) for param in self.parameters])
        self.add_updates(self.attribution_updates)

    def _compute_jacobians(self):
        if self.components is None or self.components.ndim == 0:
            raise ValueError("can't infer jacobians; no components specified")
        elif self.parameters is None or len(self.parameters) == 0:
            raise ValueError("can't infer jacobians; no parameters specified")
        logging.info("Taking the component jacobians")
        jacobians = OrderedDict(
             equizip(self.parameters, gradient.jacobian(
                 self.components, self.parameters)))
        logging.info("The component jacobian computation graph is built")
        return jacobians


class LeNet(FeedforwardSequence, Initializable):
    """LeNet-like convolutional network.

    The class implements LeNet, which is a convolutional sequence with
    an MLP on top (several fully-connected layers). For details see
    [LeCun95]_.

    .. [LeCun95] LeCun, Yann, et al.
       *Comparison of learning algorithms for handwritten digit
       recognition.*,
       International conference on artificial neural networks. Vol. 60.

    Parameters
    ----------
    conv_activations : list of :class:`.Brick`
        Activations for convolutional network.
    num_channels : int
        Number of channels in the input image.
    image_shape : tuple
        Input image shape.
    filter_sizes : list of tuples
        Filter sizes of :class:`.blocks.conv.ConvolutionalLayer`.
    feature_maps : list
        Number of filters for each of convolutions.
    pooling_sizes : list of tuples
        Sizes of max pooling for each convolutional layer.
    top_mlp_activations : list of :class:`.blocks.bricks.Activation`
        List of activations for the top MLP.
    top_mlp_dims : list
        Numbers of hidden units and the output dimension of the top MLP.
    conv_step : tuples
        Step of convolution (similar for all layers).
    border_mode : str
        Border mode of convolution (similar for all layers).

    """
    def __init__(self, conv_activations, num_channels, image_shape,
                 filter_sizes, feature_maps, pooling_sizes,
                 top_mlp_activations, top_mlp_dims,
                 conv_step=None, border_mode='valid',
                 tied_biases=True, **kwargs):
        if conv_step is None:
            self.conv_step = (1, 1)
        else:
            self.conv_step = conv_step
        self.num_channels = num_channels
        self.image_shape = image_shape
        self.top_mlp_activations = top_mlp_activations
        self.top_mlp_dims = top_mlp_dims
        self.border_mode = border_mode
        self.tied_biases = tied_biases

        conv_parameters = zip(filter_sizes, feature_maps)

        # Construct convolutional layers with corresponding parameters
        self.layers = list(interleave([
            (Convolutional(filter_size=filter_size,
                           num_filters=num_filter,
                           step=self.conv_step,
                           border_mode=self.border_mode,
                           tied_biases=self.tied_biases,
                           name='conv_{}'.format(i))
             for i, (filter_size, num_filter)
             in enumerate(conv_parameters)),
            conv_activations,
            (MaxPooling(size, name='pool_{}'.format(i))
             for i, size in enumerate(pooling_sizes))]))

        self.conv_sequence = ConvolutionalSequence(self.layers, num_channels,
                                                   image_size=image_shape)

        # Construct a top MLP
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims)

        # We need to flatten the output of the last convolutional layer.
        # This brick accepts a tensor of dimension (batch_size, ...) and
        # returns a matrix (batch_size, features)
        self.flattener = Flattener()
        application_methods = [self.conv_sequence.apply, self.flattener.apply,
                               self.top_mlp.apply]
        super(LeNet, self).__init__(application_methods, **kwargs)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')

        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [np.prod(conv_out_dim)] + self.top_mlp_dims


def main(save_to, num_epochs, feature_maps=None, mlp_hiddens=None,
         conv_sizes=None, pool_sizes=None, batch_size=500,
         num_batches=None):
    if feature_maps is None:
        feature_maps = [6, 16]
    if mlp_hiddens is None:
        mlp_hiddens = [120, 84]
    if conv_sizes is None:
        conv_sizes = [5, 5]
    if pool_sizes is None:
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
                    border_mode='full',
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
    x = tensor.tensor4('features')
    y = tensor.lmatrix('targets')

    # Normalize input and apply the convnet
    probs = convnet.apply(x)
    cost = (CategoricalCrossEntropy().apply(y.flatten(), probs)
            .copy(name='cost'))
    components = (ComponentwiseCrossEntropy().apply(y.flatten(), probs)
            .copy(name='components'))
    error_rate = (MisclassificationRate().apply(y.flatten(), probs)
                  .copy(name='error_rate'))

    cg = ComputationGraph([cost, error_rate, components])

    # Apply regularization to the cost
    weights = VariableFilter(roles=[WEIGHT])(cg.variables)
    cost = cost + sum([0.005 * (W ** 2).sum() for W in weights])
    cost.name = 'cost_with_regularization'

    mnist_train = MNIST(("train",))
    mnist_train_stream = DataStream.default_stream(
        mnist_train, iteration_scheme=ShuffledScheme(
            mnist_train.num_examples, batch_size))

    mnist_test = MNIST(("test",))
    mnist_test_stream = DataStream.default_stream(
        mnist_test,
        iteration_scheme=ShuffledScheme(
            mnist_test.num_examples, batch_size))

    # Train with simple SGD
    algorithm = AttributedGradientDescent(
        cost=cost, parameters=cg.parameters, components=components,
        components_size=output_size,
        step_rule=Scale(learning_rate=0.1))

    # `Timing` extension reports time for reading data, aggregating a batch
    # and monitoring;
    # `ProgressBar` displays a nice progress bar during training.
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs,
                              after_n_batches=num_batches),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      mnist_test_stream,
                      prefix="test"),
                  TrainingDataMonitoring(
                      [cost, error_rate,
                       aggregation.mean(algorithm.total_gradient_norm)],
                      prefix="train",
                      after_epoch=True),
    #              Checkpoint(save_to),
                  ProgressBar(),
                  Printing()]

    model = Model(cost)

    main_loop = MainLoop(
        algorithm,
        mnist_train_stream,
        model=model,
        extensions=extensions)

    main_loop.run()
    param, hist = zip(*algorithm.attributions.items())
    for pindex in range(0, len(hist)):
        allvals = hist[pindex].get_value()
        pvals = param[pindex].get_value()
        if np.prod(allvals.shape[1:]) <= 700:
            allvals = np.reshape(allvals, (allvals.shape[0], 1, -1))
            pvals = np.reshape(pvals, (1, -1))
        elif (hist[pindex].tag.for_parameter.name == 'W' and isinstance(
              hist[pindex].tag.for_parameter.tag.annotations[0], Linear)):
            allvals = np.transpose(allvals, (0, 2, 1))
            pvals = np.transpose(pvals, (1, 0))
        else:
            allvals = np.reshape(allvals, allvals.shape[0:2] + (-1,))
            pvals = np.reshape(pvals, (pvals.shape[0], -1))
        for nindex in range(0, allvals.shape[1]):
            vals = allvals[:,nindex,:]
            name = ('unit %d' % nindex) if allvals.shape[1] > 1 else 'units'
            print('Attribution for parameter %s for layer %s %s' % (
                hist[pindex].tag.for_parameter.name,
                hist[pindex].tag.for_parameter.tag.annotations[0].name,
                name))
            svals = np.sort(vals, axis=0).reshape((vals.shape[0], -1))
            sinds = np.argsort(vals, axis=0).reshape((vals.shape[0], -1))
            for j in range(svals.shape[1]):
                print('Sorted hist for weight', j, pvals[nindex, j])
                limit = max(abs(svals[:,j]))
                for k in range(svals.shape[0]):
                    n = int(np.nan_to_num(32 * svals[k, j] / limit))
                    if n < 0:
                        s = (32 + n) * ' ' + (-n) * '#'
                    else:
                        s = 32 * ' ' + (n + 1) * '#'
                    print(s, svals[k, j], sinds[k, j])

            bounds = sorted(zip(
                vals.argmin(axis=0).flatten(),
                vals.argmax(axis=0).flatten()))
            bc = defaultdict(int)
            for b in bounds:
                if b[0] != b[1]:
                    bc[b] += 1
            for x in range(10):
                printed = False
                for y in range(10):
                    amt = bc[(x, y)]
                    if amt:
                        print('%d -> %d:%s %d' %
                            (y, x, '#' * int(4 * np.log2(amt)), amt))
                        printed = True
                if printed:
                    print()

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training a convolutional network "
                            "on the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=5,
                        help="Number of training epochs to do.")
    parser.add_argument("save_to", default="mnist.pkl", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    parser.add_argument("--feature-maps", type=int, nargs='+',
                        default=[6, 16], help="List of feature maps numbers.")
    parser.add_argument("--mlp-hiddens", type=int, nargs='+', default=[120, 84],
                        help="List of numbers of hidden units for the MLP.")
    parser.add_argument("--conv-sizes", type=int, nargs='+', default=[5, 5],
                        help="Convolutional kernels sizes. The kernels are "
                        "always square.")
    parser.add_argument("--pool-sizes", type=int, nargs='+', default=[2, 2],
                        help="Pooling sizes. The pooling windows are always "
                             "square. Should be the same length as "
                             "--conv-sizes.")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Batch size.")
    args = parser.parse_args()
    main(**vars(args))
