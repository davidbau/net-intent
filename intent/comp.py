"""Convolutional network example.

Run the training for 20 epochs with
```
python run.py --num-epochs 20
```

"""
import logging
from argparse import ArgumentParser

from theano import tensor

from blocks.algorithms import Scale, AdaDelta, GradientDescent
from blocks.bricks import Activation
from blocks.bricks import Linear
from blocks.bricks.conv import Convolutional
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint, Load
from blocks.filter import VariableFilter
from blocks.filter import get_brick
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.monitoring.evaluators import AggregationBuffer
from blocks.roles import WEIGHT, BIAS
from blocks.roles import OUTPUT
from blocks.serialization import load
from fuel.datasets import MNIST
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from intent.lenet import LeNet, create_lenet_5
from intent.actpic import ActpicExtension
from intent.synpic import SynpicExtension, CasewiseCrossEntropy
from collections import OrderedDict
from filmstrip import Filmstrip
from filmstrip import plan_grid
import numpy
import pickle


class SaveImages(SimpleExtension):
    def __init__(self, picsources=None, pattern=None,
            title=None, data=None, graph=None, graph_len=None,
            unit_order=None, **kwargs):
        kwargs.setdefault("before_training", True)
        self.picsources = picsources
        self.count = 0
        if pattern is None:
            pattern = 'pics/syn/%s_%04d.jpg'
        self.pattern = pattern
        self.unit_order = unit_order
        self.title = title
        self.data = data
        self.graph = graph
        self.graph_len = graph_len
        self.graph_data = None
        # Now create an AggregationBuffer for theano variables to monitor
        self.variables = AggregationBuffer(data, use_take_last=True)
        super(SaveImages, self).__init__(**kwargs)

    def do(self, callback_name, *args):
        self.parse_args(callback_name, args)
        if (callback_name == 'before_training'):
            self.main_loop.algorithm.add_updates(
                    self.variables.accumulation_updates)
            self.variables.initialize_aggregators()
        else:
            title = self.title
            if self.data:
                values = dict((k, float(v))
                    for k, v in self.variables.get_aggregated_values().items())
                values['i'] = self.count
                title = title.format(**values)
            graph = None
            if self.graph:
                if self.graph_data is None:
                    self.graph_data = numpy.array(())
                self.graph_data = numpy.append(
                        self.graph_data, values[self.graph])
                graph = self.graph_data / self.graph_data.max()
            filename = self.pattern % ('composite', self.count)
            self.count += 1
            picdata = [ps.get_picdata() for ps in self.picsources]
            self.save_composite_image(
                    title=title, graph=graph, graph_len=self.graph_len,
                    picdata=picdata, unit_order=self.unit_order,
                    filename=filename, aspect_ratio=16.0/9.0)
            self.variables.initialize_aggregators()

    def save_composite_image(self,
                title=None, graph=None, graph_len=None, picdata=None,
                filename=None, aspect_ratio=None, unit_order=None):
        if filename is None:
            pattern = 'synpic.jpg'
        unit_count = 0
        layer_count = 0
        if graph is not None:
            unit_count += 4 # TODO: make configurable
        if title is not None:
            unit_count += 1
        merged = OrderedDict([
            (k, [d[k] for d in picdata]) for k in picdata[0].keys()])
        unit_width = 0
        for name, d in merged.items():
            for dat in d:
                if len(dat.shape) != 4:
                    raise NotImplementedError('%s has %s dimensions' % (
                        name, dat.shape))
                unit_count += dat.shape[0]
                unit_width = max(unit_width, dat.shape[1])
            layer_count += 1
        unit_width += 1
        column_height, column_count = plan_grid(unit_count + layer_count,
                aspect_ratio, dat.shape[-2:], (1, unit_width))
        filmstrip = Filmstrip(image_shape=dat.shape[-2:],
            grid_shape=(column_height, column_count * unit_width))
        pos = 0
        if graph is not None:
            col, row = divmod(pos, column_height)
            filmstrip.plot_graph((row, col * unit_width + 1),
                (4, unit_width - 1),
                graph, graph_len)
            pos += 4
        if title is not None:
            col, row = divmod(pos, column_height)
            filmstrip.set_text((row, col * unit_width + unit_width // 2),
                    title)
            pos += 1
        for layername, d in merged.items():
            units = d[0].shape[0]
            col, row = divmod(pos, column_height)
            filmstrip.set_text((row, col * unit_width + unit_width // 2),
                    layername)
            pos += 1
            if unit_order:
                ordering = unit_order[layername]
            else:
                ordering = range(units)
            scales = [dat.std() * 5 for dat in d]
            for unit in ordering:
                for dat, scale in zip(d, scales):
                    col, row = divmod(pos, column_height)
                    filmstrip.set_text((row, col * unit_width), "%d:" % unit)
                    im = dat[unit, :, :, :]
                    # imin = im.min()
                    # imax = im.max()
                    # scale = (imax - imin) * 0.7
                    im = im / (scale + 1e-9) + 0.5
                    for label in range(im.shape[0]):
                        filmstrip.set_image((row, label + 1 +
                            col * unit_width), im[label, :, :])
                    pos += 1
        filmstrip.save(filename)

def argsort(seq):
    # http://stackoverflow.com/questions/3382352#3382369
    return sorted(range(len(seq)), key=seq.__getitem__)

def compute_unit_order(data):
    result = {}
    for (layername, paramname), hist in data.items():
        if paramname != 'b':
            continue
        result[layername] = argsort(
                list(tuple(r) for r in hist.transpose().argsort(axis=1)))
    return result

def main(save_to, num_epochs, resume=False, **kwargs):
    if resume:
        with open(save_to, 'rb') as source:
            main_loop = load(source)
    else:
        main_loop = create_main_loop(save_to, num_epochs, **kwargs)

    if main_loop.status['epochs_done'] < num_epochs:
        main_loop.run()

def create_main_loop(save_to, num_epochs, unit_order=None,
        batch_size=500, num_batches=None):
    image_size = (28, 28)
    output_size = 10
    convnet = create_lenet_5()
    x = tensor.tensor4('features')
    y = tensor.lmatrix('targets')

    # Normalize input and apply the convnet
    probs = convnet.apply(x)
    case_costs = CasewiseCrossEntropy().apply(y.flatten(), probs)
    cost = case_costs.mean().copy(name='cost')
    # cost = (CategoricalCrossEntropy().apply(y.flatten(), probs)
    #         .copy(name='cost'))
    error_rate = (MisclassificationRate().apply(y.flatten(), probs)
                  .copy(name='error_rate'))

    cg = ComputationGraph([cost, error_rate])

    # Apply regularization to the cost
    weights = VariableFilter(roles=[WEIGHT])(cg.variables)
    cost = cost + sum([0.0003 * (W ** 2).sum() for W in weights])
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

    # Generate pics for biases
    biases = VariableFilter(roles=[BIAS])(cg.parameters)

    # Train with simple SGD
    algorithm = GradientDescent(
        cost=cost,
        parameters=cg.parameters,
        step_rule=AdaDelta())

    # Find layer outputs to probe
    outs = OrderedDict(reversed(list((get_brick(out).name, out)
            for out in VariableFilter(
                roles=[OUTPUT], bricks=[Convolutional, Linear])(
                    cg.variables))))

    actpic_extension = ActpicExtension(
        actpic_variables=outs,
        case_labels=y,
        pics=x,
        label_count=output_size,
        rectify=True,
        data_stream=mnist_test_stream,
        after_batch=True)

    synpic_extension = SynpicExtension(
        synpic_parameters=biases,
        case_costs=case_costs,
        case_labels=y,
        pics=x,
        batch_size=batch_size,
        pic_size=image_size,
        label_count=output_size,
        after_batch=True)

    # Impose an orderint for the SaveImages extension
    if unit_order is not None:
        with open(unit_order, 'rb') as handle:
            histograms = pickle.load(handle)
        unit_order = compute_unit_order(histograms)

    # `Timing` extension reports time for reading data, aggregating a batch
    # and monitoring;
    # `ProgressBar` displays a nice progress bar during training.
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs,
                              after_n_batches=num_batches),
                  actpic_extension,
                  synpic_extension,
                  SaveImages(picsources=[synpic_extension, actpic_extension],
                      title="LeNet-5: batch {i}, " +
                          "cost {cost_with_regularization:.2f}, " + 
                          "trainerr {error_rate:.3f}",
                      data=[cost, error_rate],
                      graph='error_rate',
                      graph_len=500,
                      unit_order=unit_order,
                      after_batch=True),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      mnist_test_stream,
                      prefix="test"),
                  TrainingDataMonitoring(
                      [cost, error_rate,
                       aggregation.mean(algorithm.total_gradient_norm)],
                      prefix="train",
                      after_epoch=True),
                  Checkpoint(save_to),
                  ProgressBar(),
                  Printing()]
    model = Model(cost)
    main_loop = MainLoop(
        algorithm,
        mnist_train_stream,
        model=model,
        extensions=extensions)

    return main_loop

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training a convolutional network "
                            "on the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=5,
                        help="Number of training epochs to do.")
    parser.add_argument("save_to", default="mnist.tar", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Batch size.")
    parser.add_argument("--unit-order", nargs="?", default=None,
                        help="Render unit ordering based on these histograms.")
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.set_defaults(resume=False)
    args = parser.parse_args()
    main(**vars(args))
