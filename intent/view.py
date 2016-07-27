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
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from intent.lenet import LeNet
from intent.maxact import MaximumActivationSearch
from intent.filmstrip import Filmstrip
from intent.rf import make_mask
from intent.rf import layerarray_fieldmap
import numpy

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

    algorithm = MaximumActivationSearch(outputs=outs)

    # Use the mnist test set, unshuffled
    mnist_test = MNIST(("test",), sources=['features'])
    mnist_test_stream = DataStream.default_stream(
        mnist_test,
        iteration_scheme=SequentialScheme(
            mnist_test.num_examples, batch_size))

    extensions = [Timing(),
                  FinishAfter(after_n_epochs=1),
                  DataStreamMonitoring(
                      [],
                      mnist_test_stream,
                      prefix="test"),
                  Checkpoint("maxact.tar"),
                  ProgressBar(),
                  Printing()]

    main_loop = MainLoop(
        algorithm,
        mnist_test_stream,
        model=model,
        extensions=extensions)

    main_loop.run()

    examples = mnist_test.get_example_stream()
    example = examples.get_data(0)[0]
    layers = convnet.layers
    for output, record in algorithm.maximum_activations.items():
        layer = get_brick(output)
        activations, indices, snapshots = (
                r.get_value() if r else None for r in record[1:])
        filmstrip = Filmstrip(
            example.shape[-2:], (indices.shape[1], indices.shape[0]),
            background='blue')
        if layer in layers:
            fieldmap = layerarray_fieldmap(layers[0:layers.index(layer) + 1])
            for unit in range(indices.shape[1]):
                for index in range(100):
                    mask = make_mask(example.shape[-2:], fieldmap, numpy.clip(
                        snapshots[index, unit, :, :], 0, numpy.inf))
                    imagenum = indices[index, unit, 0]
                    filmstrip.set_image((unit, index),
                            examples.get_data(imagenum)[0], mask)
        else:
            for unit in range(indices.shape[1]):
                for index in range(100):
                    imagenum = indices[index, unit]
                    filmstrip.set_image((unit, index),
                            examples.get_data(imagenum)[0])
        filmstrip.save(layer.name + '_maxact.jpg')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("Visualization for the MNIST dataset.")
    parser.add_argument("save_to", default="noisy-mnist.tar", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    args = parser.parse_args()
    main(**vars(args))
