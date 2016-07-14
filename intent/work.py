"""Convolutional network example.

Run the training for 20 epochs with
```
python run.py --num-epochs 20
```

"""
import logging
from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, HTTPServer

from theano import tensor

from blocks.bricks.conv import Convolutional
from blocks.bricks import application
from blocks.bricks import Brick
from blocks.bricks import Linear
from blocks.bricks import Rectifier
from blocks.bricks import Activation
from blocks.bricks import Softmax
from blocks.bricks.cost import CategoricalCrossEntropy
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
from blocks.monitoring.aggregation import AggregationScheme
from blocks.monitoring.aggregation import Aggregator
from blocks.serialization import load_parameters
from blocks.utils import dict_subset
from blocks.utils import shared_floatx
from blocks.utils import shared_like
from collections import OrderedDict
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from intent.ablation import ConfusionMatrix
from intent.ablation import ConfusionImage
from intent.ablation import Sum
from intent.ablation import ablate_inputs
from intent.lenet import create_lenet_5
from intent.maxact import MaximumActivationSearch
from intent.filmstrip import Filmstrip
from intent.rf import make_mask
from intent.rf import layerarray_fieldmap
from prior import create_fair_basis
from theano import gradient
from theano import tensor
from theano.ifelse import ifelse
from theano.printing import Print
import theano
import numpy
import numbers

# For testing
from blocks.roles import OUTPUT, BIAS

class Concatenate(AggregationScheme):
    """Aggregation scheme which concatenates results on axis 0.
    Parameters
    ----------
    variable : :class:`~tensor.TensorVariable`
        Theano variable for the quantity to be concatenated
    """
    def __init__(self, variable):
        self.variable = variable

    def get_aggregator(self):
        initialized = shared_like(0.)
        total_acc = shared_like(self.variable)
        empty_init = tensor.zeros(
                (0,) * self.variable.ndim, dtype=self.variable.dtype)

        conditional_update_num = tensor.concatenate([
            ifelse(initialized, total_acc, empty_init),
            self.variable])

        initialization_updates = [(total_acc, empty_init),
                                  (initialized, tensor.zeros_like(initialized))]

        accumulation_updates = [(total_acc, conditional_update_num),
                                (initialized, tensor.ones_like(initialized))]

        aggregator = Aggregator(aggregation_scheme=self,
                                initialization_updates=initialization_updates,
                                accumulation_updates=accumulation_updates,
                                readout_variable=(total_acc))

        return aggregator

class SensitiveUnitCount(Brick):
    """Counts number of units (or parameters) with nonzero gradient for
    each instance.
    """
    @application(outputs=["sensitive_unit_count"])
    def apply(self, y, y_hat, biases):
        cost = tensor.nnet.categorical_crossentropy(y_hat, y.flatten())
        predicted = y_hat.argmax(axis=1)
        # Here we just count the number of unit biases with nonzero gradient
        jacobians = gradient.jacobian(cost, biases)
        counts = tensor.zeros_like(y)
        for j in jacobians:
            counts += tensor.neq(j, 0).sum(axis=1)
        return counts

class ActiveUnitCount(Brick):
    """Counts number of units with nonzero activation for each instance.
    """
    @application(outputs=["active_unit_count"])
    def apply(self, outs):
        counts = None
        for o in outs:
            while o.ndim > 2:
                o = o.max(axis=o.ndim - 1)
            count = tensor.gt(o, 0).sum(axis=1)
            if counts is None:
                counts = count
            else:
                counts += count
        return counts

class IgnoredUnitCount(Brick):
    """Counts number of units (or parameters) with nonzero gradient for
    each instance.
    """
    @application(outputs=["sensitive_unit_count"])
    def apply(self, y, y_hat, biases, outs):
        cost = tensor.nnet.categorical_crossentropy(y_hat, y.flatten())
        predicted = y_hat.argmax(axis=1)
        # Here we just count the number of unit biases with nonzero gradient
        jacobians = gradient.jacobian(cost, biases)
        counts = tensor.zeros_like(y)
        for j, o in zip(jacobians, outs):
            while o.ndim > 2:
                o = o.max(axis=o.ndim - 1)
            counts += (tensor.gt(o, 0) * tensor.eq(j, 0)).sum(axis=1)
        return counts

class WorkRater:
    def __init__(self, save_to):
        batch_size = 500
        image_size = (28, 28)
        output_size = 10
        convnet = create_lenet_5()
        layers = convnet.layers

        mnist_test = MNIST(("test",), sources=['features', 'targets'])

        x = tensor.tensor4('features')
        y = tensor.lmatrix('targets')

        # Normalize input and apply the convnet
        probs = convnet.apply(x)
        cg = ComputationGraph([probs])

        def full_brick_name(brick):
            return '/'.join([''] + [b.name for b in brick.get_unique_path()])

        # Find layer outputs to probe
        outmap = OrderedDict((full_brick_name(get_brick(out)), out)
                for out in VariableFilter(
                    roles=[OUTPUT], bricks=[Convolutional, Linear])(
                        cg.variables))
        # Generate pics for biases
        biases = VariableFilter(roles=[BIAS])(cg.parameters)

        # Generate parallel array, in the same order, for outputs
        outs = [outmap[full_brick_name(get_brick(b))] for b in biases]

        # Figure work count
        error_rate = (MisclassificationRate().apply(y.flatten(), probs)
                      .copy(name='error_rate'))
        sensitive_unit_count = (SensitiveUnitCount().apply(
                y.flatten(), probs, biases).copy(name='sensitive_unit_count'))
        sensitive_unit_count.tag.aggregation_scheme = (
                Concatenate(sensitive_unit_count))
        active_unit_count = (ActiveUnitCount().apply(
                outs).copy(name='active_unit_count'))
        active_unit_count.tag.aggregation_scheme = (
                Concatenate(active_unit_count))
        ignored_unit_count = (IgnoredUnitCount().apply(y.flatten(),
                probs, biases, outs).copy(name='ignored_unit_count'))
        ignored_unit_count.tag.aggregation_scheme = (
                Concatenate(ignored_unit_count))

        model = Model([
            error_rate,
            sensitive_unit_count,
            active_unit_count,
            ignored_unit_count])

        # Load it with trained parameters
        params = load_parameters(open(save_to, 'rb'))
        model.set_parameter_values(params)

        mnist_test = MNIST(("test",))
        mnist_test_stream = DataStream.default_stream(
            mnist_test,
            iteration_scheme=SequentialScheme(
                mnist_test.num_examples, batch_size))

        evaluator = DatasetEvaluator([
            error_rate,
            sensitive_unit_count,
            active_unit_count,
            ignored_unit_count
            ])
        results = evaluator.evaluate(mnist_test_stream)

        def save_ranked_image(scores, filename):
            sorted_instances = scores.argsort()
            filmstrip = Filmstrip(image_shape=(28, 28), grid_shape=(100, 100))
            for i, index in enumerate(sorted_instances):
                filmstrip.set_image((i // 100, i % 100),
                        mnist_test.get_data(request=index)[0])
            filmstrip.save(filename)

        save_ranked_image(results['sensitive_unit_count'], 'sensitive.jpg')
        save_ranked_image(results['active_unit_count'], 'active.jpg')
        save_ranked_image(results['ignored_unit_count'], 'ignored.jpg')
        # import pdb; pdb.set_trace()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("Gradient descent vis for the MNIST dataset.")
    parser.add_argument("save_to", default="mnist.tar", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    args = parser.parse_args()
    WorkRater(**vars(args))
