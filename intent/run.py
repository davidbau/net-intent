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
from blocks.bricks import Rectifier
from blocks.bricks import Activation
from blocks.bricks import Softmax
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint, Load
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.roles import WEIGHT
from fuel.datasets import MNIST
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from intent.lenet import LeNet, create_lenet_5
from intent.attrib import AttributionExtension
from intent.attrib import ComponentwiseCrossEntropy
from intent.attrib import print_attributions
from intent.attrib import save_attributions
from intent.ablation import ConfusionMatrix
from intent.ablation import Sum

# For testing

def main(save_to, num_epochs,
         regularization=0.0003, subset=None, num_batches=None,
         histogram=None, resume=False):
    batch_size = 500
    output_size = 10
    convnet = create_lenet_5()
    layers = convnet.layers

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
    confusion = (ConfusionMatrix().apply(y.flatten(), probs)
                  .copy(name='confusion'))
    confusion.tag.aggregation_scheme = Sum(confusion)

    cg = ComputationGraph([cost, error_rate, components])

    # Apply regularization to the cost
    weights = VariableFilter(roles=[WEIGHT])(cg.variables)
    cost = cost + sum([regularization * (W ** 2).sum() for W in weights])
    cost.name = 'cost_with_regularization'

    if subset:
        mnist_train = MNIST(("train",), subset=slice(1, subset))
    else:
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
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=AdaDelta())

    # `Timing` extension reports time for reading data, aggregating a batch
    # and monitoring;
    # `ProgressBar` displays a nice progress bar during training.
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs,
                              after_n_batches=num_batches),
                  DataStreamMonitoring(
                      [cost, error_rate, confusion],
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

    if histogram:
        attribution = AttributionExtension(
            components=components,
            parameters=cg.parameters,
            components_size=output_size,
            after_batch=True)
        extensions.insert(0, attribution)

    if resume:
        extensions.append(Load(save_to, True, True))

    model = Model(cost)

    main_loop = MainLoop(
        algorithm,
        mnist_train_stream,
        model=model,
        extensions=extensions)

    main_loop.run()

    if histogram:
        save_attributions(attribution, filename=histogram)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training a convolutional network "
                            "on the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=5,
                        help="Number of training epochs to do.")
    parser.add_argument("--histogram", help="histogram file")
    parser.add_argument("save_to", default="mnist.tar", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    parser.add_argument('--regularization', type=float, default=0.0003,
                        help="Amount of regularization to apply.")
    parser.add_argument('--subset', type=int, default=None,
                        help="Size of limited training set.")
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.set_defaults(resume=False)
    args = parser.parse_args()
    main(**vars(args))
