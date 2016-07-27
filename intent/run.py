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
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint, Load
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.graph.bn import batch_normalization
from blocks.graph.bn import get_batch_normalization_updates
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.roles import WEIGHT, BIAS
from fuel.datasets import MNIST
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from intent.lenet import LeNet, create_lenet_5
from intent.noisy import NoisyLeNet, create_noisy_lenet_5
from intent.noisy import NITS, NOISE, NoiseExtension
from intent.noisy import NoisyDataStreamMonitoring
from intent.attrib import AttributionExtension
from intent.attrib import ComponentwiseCrossEntropy
from intent.attrib import print_attributions
from intent.attrib import save_attributions
from intent.ablation import ConfusionMatrix
from intent.ablation import Sum

# For testing

def main(save_to, num_epochs,
         num_batches=None, resume=False, histogram=None):
    batch_size = 500
    output_size = 10
    convnet = create_noisy_lenet_5(batch_size)
    layers = convnet.layers

    mnist_test = MNIST(("test",), sources=['features', 'targets'])

    x = tensor.tensor4('features')
    y = tensor.lmatrix('targets')

    # Normalize input and apply the convnet
    test_probs = convnet.apply(x)
    with batch_normalization(convnet):
        train_probs = convnet.apply(x)

    test_cost = (CategoricalCrossEntropy().apply(y.flatten(), test_probs)
            .copy(name='cost'))
    test_components = (ComponentwiseCrossEntropy().apply(y.flatten(),
            test_probs).copy(name='components'))
    test_error_rate = (MisclassificationRate().apply(y.flatten(), test_probs)
            .copy(name='error_rate'))
    test_confusion = (ConfusionMatrix().apply(y.flatten(), test_probs)
            .copy(name='confusion'))
    test_confusion.tag.aggregation_scheme = Sum(test_confusion)

    train_cost = (CategoricalCrossEntropy().apply(y.flatten(), train_probs)
            .copy(name='cost'))
    train_error_rate = (MisclassificationRate().apply(y.flatten(), train_probs)
            .copy(name='error_rate'))

    # Apply regularization to the cost
    test_cg = ComputationGraph([test_cost, test_error_rate, test_components])
    test_nits = VariableFilter(roles=[NITS])(test_cg.auxiliary_variables)
    test_nit_rate = tensor.concatenate([n.flatten() for n in test_nits]).mean()
    test_nit_rate.name = 'nit_rate'
    test_cost = test_cost + test_nit_rate
    test_cost.name = 'cost_with_regularization'

    # Apply regularization to the cost
    train_cg = ComputationGraph([train_cost])
    train_nits = VariableFilter(roles=[NITS])(train_cg.auxiliary_variables)
    train_nit_rate = tensor.concatenate([n.flatten() for n in train_nits]
            ).mean()
    train_nit_rate.name = 'nit_rate'
    train_cost = train_cost + train_nit_rate
    train_cost.name = 'cost_with_regularization'

    # Do batchnorm updates
    alpha = 0.1
    pop_updates = get_batch_normalization_updates(train_cg)
    bn_updates = [(p, m * alpha + p * (1 - alpha)) for p, m in pop_updates]

    mnist_train = MNIST(("train",))
    mnist_train_stream = DataStream.default_stream(
        mnist_train, iteration_scheme=ShuffledScheme(
            mnist_train.num_examples, batch_size))

    mnist_test = MNIST(("test",))
    mnist_test_stream = DataStream.default_stream(
        mnist_test,
        iteration_scheme=ShuffledScheme(
            mnist_test.num_examples, batch_size))

    trainable_parameters = VariableFilter(roles=[WEIGHT, BIAS])(
            train_cg.parameters)
    noise_parameters = VariableFilter(roles=[NOISE])(
            train_cg.parameters)

    # Train with simple SGD
    # from theano.compile.nanguardmode import NanGuardMode

    algorithm = GradientDescent(
        cost=train_cost,
        parameters=trainable_parameters,
        step_rule=AdaDelta())

    algorithm.add_updates(bn_updates)

    add_noise = NoiseExtension(
        noise_parameters=noise_parameters)

    # `Timing` extension reports time for reading data, aggregating a batch
    # and monitoring;
    # `ProgressBar` displays a nice progress bar during training.
    extensions = [add_noise,
                  Timing(),
                  FinishAfter(after_n_epochs=num_epochs,
                              after_n_batches=num_batches),
                  NoisyDataStreamMonitoring(
                      [test_cost, test_error_rate, test_nit_rate,
                          test_confusion],
                      mnist_test_stream,
                      noise_parameters=noise_parameters,
                      prefix="test"),
                  TrainingDataMonitoring(
                      [train_cost, train_error_rate, train_nit_rate,
                       aggregation.mean(algorithm.total_gradient_norm)],
                      prefix="train",
                      after_batch=True),
                  Checkpoint(save_to),
                  ProgressBar(),
                  Printing()]

    if histogram:
        attribution = AttributionExtension(
            components=components,
            parameters=trainable_parameters,
            components_size=output_size,
            after_batch=True)
        extensions.insert(0, attribution)

    if resume:
        extensions.append(Load(save_to, True, True))

    model = Model(train_cost)

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
    parser.add_argument("save_to", default="noisy-mnist.tar", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    parser.add_argument("--histogram", help="histogram file")
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.set_defaults(resume=False)
    args = parser.parse_args()
    main(**vars(args))
