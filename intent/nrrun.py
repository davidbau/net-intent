"""Convolutional network example, using Noisy ResNet and CIFAR-10.

"""
import logging
from argparse import ArgumentParser

import theano
from theano import tensor

from blocks.algorithms import Scale, AdaDelta, GradientDescent, Momentum
from blocks.algorithms import CompositeRule, Restrict
from blocks.bricks import Rectifier
from blocks.bricks import Activation
from blocks.bricks import Softmax
from blocks.bricks import Linear
from blocks.bricks.conv import Convolutional
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint, Load
from blocks.filter import VariableFilter
from blocks.filter import get_brick
from blocks.graph import batch_normalization
from blocks.graph import get_batch_normalization_updates
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.roles import BIAS
from blocks.roles import WEIGHT
from blocks.roles import OUTPUT
from blocks_extras.extensions.plot import Plot  
from fuel.datasets import CIFAR10
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers.image import RandomFixedSizeCrop
from intent.resnet import create_res_net
from intent.attrib import ComponentwiseCrossEntropy
from intent.attrib import print_attributions
from intent.attrib import save_attributions
from intent.ablation import ConfusionMatrix
from intent.ablation import Sum
from intent.noisy import NITS, NOISE, LOG_SIGMA, NoiseExtension
from intent.noisy import NoisyDataStreamMonitoring
from intent.transform import RandomFlip
from intent.transform import RandomPadCropFlip
from intent.transform import NormalizeBatchLevels
from intent.schedule import EpochSchedule
import json
from json import JSONEncoder, dumps
import numpy

# Increase recursion limit for pickle to work on a big resnet
import sys
sys.setrecursionlimit(50000)


# For testing

def main(save_to, num_epochs,
         weight_decay=0.0001, noise_pressure=0.0001, subset=None, num_batches=None,
         batch_size=None, histogram=None, resume=False):
    output_size = 10
    convnet = create_res_net(mid_noise=True, noise_batch_size=batch_size)

    x = tensor.tensor4('features')
    y = tensor.lmatrix('targets')

    # Normalize input and apply the convnet
    test_probs = convnet.apply(x)
    test_cost = (CategoricalCrossEntropy().apply(y.flatten(), test_probs)
            .copy(name='cost'))
    test_error_rate = (MisclassificationRate().apply(y.flatten(), test_probs)
                  .copy(name='error_rate'))
    test_confusion = (ConfusionMatrix().apply(y.flatten(), test_probs)
                  .copy(name='confusion'))
    test_confusion.tag.aggregation_scheme = Sum(test_confusion)

    test_cg = ComputationGraph([test_cost, test_error_rate])

    # Apply dropout to all layer outputs except final softmax
    # dropout_vars = VariableFilter(
    #         roles=[OUTPUT], bricks=[Convolutional],
    #         theano_name_regex="^conv_[25]_apply_output$")(test_cg.variables)
    # drop_cg = apply_dropout(test_cg, dropout_vars, 0.5)

    # Apply 0.2 dropout to the pre-averaging layer
    # dropout_vars_2 = VariableFilter(
    #         roles=[OUTPUT], bricks=[Convolutional],
    #         theano_name_regex="^conv_8_apply_output$")(test_cg.variables)
    # train_cg = apply_dropout(test_cg, dropout_vars_2, 0.2)

    # Apply 0.2 dropout to the input, as in the paper
    # train_cg = apply_dropout(test_cg, [x], 0.2)
    # train_cg = drop_cg
    # train_cg = apply_batch_normalization(test_cg)

    # train_cost, train_error_rate, train_components = train_cg.outputs

    with batch_normalization(convnet):
        train_probs = convnet.apply(x)
    train_cost = (CategoricalCrossEntropy().apply(y.flatten(), train_probs)
                .copy(name='cost'))
    train_components = (ComponentwiseCrossEntropy().apply(y.flatten(),
                train_probs).copy(name='components'))
    train_error_rate = (MisclassificationRate().apply(y.flatten(),
                train_probs).copy(name='error_rate'))
    train_cg = ComputationGraph([train_cost,
                train_error_rate, train_components])
    population_updates = get_batch_normalization_updates(train_cg)
    bn_alpha = 0.9
    extra_updates = [(p, p * bn_alpha + m * (1 - bn_alpha))
                for p, m in population_updates]

    # for annealing
    nit_penalty = theano.shared(numpy.asarray(noise_pressure, dtype=theano.config.floatX))
    nit_penalty.name = 'nit_penalty'

    # Compute noise rates for test graph
    test_logsigma = VariableFilter(roles=[LOG_SIGMA])(test_cg.variables)
    test_mean_log_sigma = tensor.concatenate([n.flatten() for n in test_logsigma]).mean()
    test_mean_log_sigma.name = 'log_sigma'
    test_nits = VariableFilter(roles=[NITS])(test_cg.auxiliary_variables)
    test_nit_rate = tensor.concatenate([n.flatten() for n in test_nits]).mean()
    test_nit_rate.name = 'nit_rate'
    test_nit_regularization = nit_penalty * test_nit_rate
    test_nit_regularization.name = 'nit_regularization'

    # Compute noise rates for training graph
    train_logsigma = VariableFilter(roles=[LOG_SIGMA])(train_cg.variables)
    train_mean_log_sigma = tensor.concatenate([n.flatten() for n in train_logsigma]).mean()
    train_mean_log_sigma.name = 'mean_log_sigma'
    train_nits = VariableFilter(roles=[NITS])(train_cg.auxiliary_variables)
    train_nit_rate = tensor.concatenate([n.flatten() for n in train_nits]).mean()
    train_nit_rate.name = 'nit_rate'
    train_nit_regularization = nit_penalty * train_nit_rate
    train_nit_regularization.name = 'nit_regularization'

    # Apply regularization to the cost
    trainable_parameters = VariableFilter(roles=[WEIGHT, BIAS])(
            train_cg.parameters)
    mask_parameters = [p for p in trainable_parameters
            if get_brick(p).name == 'mask']
    noise_parameters = VariableFilter(roles=[NOISE])(train_cg.parameters)
    biases = VariableFilter(roles=[BIAS])(train_cg.parameters)
    weights = VariableFilter(roles=[WEIGHT])(train_cg.variables)
    l2_norm = sum([(W ** 2).sum() for W in weights])
    l2_norm.name = 'l2_norm'
    l2_regularization = weight_decay * l2_norm
    l2_regularization.name = 'l2_regularization'

    # testversion
    test_cost = test_cost + l2_regularization + test_nit_regularization
    test_cost.name = 'cost_with_regularization'

    # Training version of cost
    train_cost_without_regularization = train_cost
    train_cost_without_regularization.name = 'cost_without_regularization'
    train_cost = train_cost + l2_regularization + train_nit_regularization
    train_cost.name = 'cost_with_regularization'

    cifar10_train = CIFAR10(("train",))
    cifar10_train_stream = RandomPadCropFlip(
        NormalizeBatchLevels(DataStream.default_stream(
            cifar10_train, iteration_scheme=ShuffledScheme(
                cifar10_train.num_examples, batch_size)),
        which_sources=('features',)),
        (32, 32), pad=4, which_sources=('features',))

    test_batch_size = 128
    cifar10_test = CIFAR10(("test",))
    cifar10_test_stream = NormalizeBatchLevels(DataStream.default_stream(
        cifar10_test,
        iteration_scheme=ShuffledScheme(
            cifar10_test.num_examples, test_batch_size)),
        which_sources=('features',))

    momentum = Momentum(0.01, 0.9)

    # Create a step rule that doubles the learning rate of biases, like Caffe.
    # scale_bias = Restrict(Scale(2), biases)
    # step_rule = CompositeRule([scale_bias, momentum])

    # Create a step rule that reduces the learning rate of noise
    scale_mask = Restrict(Scale(1e-3), mask_parameters)
    step_rule = CompositeRule([scale_mask, momentum])

    # from theano.compile.nanguardmode import NanGuardMode

    # Train with simple SGD
    algorithm = GradientDescent(
        cost=train_cost, parameters=trainable_parameters,
        step_rule=step_rule)
    algorithm.add_updates(extra_updates)

    #,
    #    theano_func_kwargs={
    #        'mode': NanGuardMode(
    #            nan_is_error=True, inf_is_error=True, big_is_error=True)})

    # `Timing` extension reports time for reading data, aggregating a batch
    # and monitoring;
    # `ProgressBar` displays a nice progress bar during training.
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs,
                              after_n_batches=num_batches),
                  EpochSchedule(momentum.learning_rate, [
                      (0, 1e-6),   # Warm up with 0.01 learning rate
                      (10, 0.1),    # Then go back to 0.1
                      (100, 0.01),
                      (150, 0.001)
                      # (83, 0.01),  # Follow the schedule in the paper
                      # (125, 0.001)
                  ]),
                  NoiseExtension(
                      noise_parameters=noise_parameters),
                  NoisyDataStreamMonitoring(
                      [test_cost, test_error_rate, test_confusion],
                      cifar10_test_stream,
                      noise_parameters=noise_parameters,
                      prefix="test"),
                  TrainingDataMonitoring(
                      [train_cost, train_error_rate, train_nit_rate,
                       train_cost_without_regularization,
                       l2_regularization,
                       train_nit_regularization,
                       momentum.learning_rate,
                       train_mean_log_sigma,
                       aggregation.mean(algorithm.total_gradient_norm)],
                      prefix="train",
                      every_n_batches=17),
                      # after_epoch=True),
                  Plot('Training performance for ' + save_to,
                      channels=[
                          ['train_cost_with_regularization',
                           'train_cost_without_regularization',
                           'train_nit_regularization',
                           'train_l2_regularization'],
                          ['train_error_rate'],
                          ['train_total_gradient_norm'],
                          ['train_mean_log_sigma'],
                      ],
                      every_n_batches=17),
                  Plot('Test performance for ' + save_to,
                      channels=[[
                          'train_error_rate',
                          'test_error_rate',
                          ]],
                      after_epoch=True),
                  Checkpoint(save_to, use_cpickle=True),
                  ProgressBar(),
                  Printing()]

    if histogram:
        attribution = AttributionExtension(
            components=train_components,
            parameters=cg.parameters,
            components_size=output_size,
            after_batch=True)
        extensions.insert(0, attribution)

    if resume:
        extensions.append(Load(save_to, True, True))

    model = Model(train_cost)

    main_loop = MainLoop(
        algorithm,
        cifar10_train_stream,
        model=model,
        extensions=extensions)

    main_loop.run()

    if histogram:
        save_attributions(attribution, filename=histogram)

    with open('execution-log.json', 'w') as outfile:
        json.dump(main_loop.log, outfile, cls=NumpyEncoder)

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            if obj.ndim == 0:
                return obj + 0
            if obj.ndim == 1:
                return obj.tolist()
            return list([self.default(row) for row in obj])
        return JSONEncoder.default(self, obj)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training a convolutional network "
                            "on the CIFAR dataset.")
    parser.add_argument("--num-epochs", type=int, default=200,
                        help="Number of training epochs to do.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Number of training examples per minibatch.")
    parser.add_argument("--histogram", help="histogram file")
    parser.add_argument("save_to", default="cifar10-resnet-noisy.tar",
                        nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    parser.add_argument('--weight-decay', type=float, default=1e-6,
                        help="Amount of W L2 regularization to apply.")
    parser.add_argument('--noise-pressure', type=float, default=1e-11,
                        help="Amount of noise regularization to apply.")
    parser.add_argument('--subset', type=int, default=None,
                        help="Size of limited training set.")
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.set_defaults(resume=False)
    args = parser.parse_args()
    main(**vars(args))
