"""All convolutional network.

Modeled after ALL-CNN-C from "Striving for Simplicity: The All
Convolutional Net" paper by Springenberg, Dosovitskiy, Brox, Riedmiller.
https://arxiv.org/pdf/1412.6806.pdf

Details also from Nervana's code at
https://github.com/NervanaSystems/neon/blob/master/examples/cifar10_allcnn.py
"""


from blocks.bricks import application
from blocks.bricks import Brick
from blocks.bricks import FeedforwardSequence
from blocks.bricks import Initializable
from blocks.bricks import MLP
from blocks.bricks import Rectifier
from blocks.bricks import Softmax
from blocks.bricks.conv import Convolutional, ConvolutionalSequence
from blocks.bricks.conv import Flattener, MaxPooling
from blocks.initialization import Constant, Uniform, NdarrayInitialization
from blocks.initialization import IsotropicGaussian
import numpy
import theano
from toolz.itertoolz import interleave
from intent.noisy import NoisyConvolutional
from intent.noisy import NoisyConvolutional2
from intent.mask import ChannelMask


class GlobalAverageFlattener(Brick):
    """Flattens the input by applying spatial averaging for each channel.
    It may be used to pass multidimensional objects like images or feature
    maps of convolutional bricks into bricks which allow only two
    dimensional input (batch, features) like MLP.
    """
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return input_.mean(axis=list(range(2, input_.ndim)))

class HeInitialization(NdarrayInitialization):
    """Initialize parameters from an isotropic Gaussian distribution.
    Parameters
    ----------
    std : float, optional
        The standard deviation of the Gaussian distribution. Defaults to 1.
    mean : float, optional
        The mean of the Gaussian distribution. Defaults to 0
    Notes
    -----
    Be careful: the standard deviation goes first and the mean goes
    second!
    """
    def __init__(self, gain=None):
        self.gain = gain or numpy.sqrt(2)

    def generate(self, rng, shape):
        if len(shape) == 2:
            fan_in = shape[0]
        elif len(shape) > 2:
            fan_in = numpy.prod(shape[1:])
        std = self.gain * numpy.sqrt(1.0 / fan_in)
        return rng.normal(0, std, size=shape).astype(theano.config.floatX)

    def __repr__(self):
        return repr_attrs(self, 'gain')

class AllConvNet(FeedforwardSequence, Initializable):
    def __init__(self, image_shape=None, output_size=None, **kwargs):
        self.num_channels = 3
        self.image_shape = image_shape or (32, 32)
        self.output_size = output_size or 10
        conv_parameters = [
                (96, 3, 1, 'half'),
                (96, 3, 1, 'half'),
                (96, 3, 2, 'half'),
                (192, 3, 1, 'half'),
                (192, 3, 1, 'half'),
                (192, 3, 2, 'half'),
                (192, 3, 1, 'half'),
                (192, 1, 1, 'valid'),
                (10, 1, 1, 'valid')
        ]
        fc_layer = 10

        self.convolutions = list([
            Convolutional(filter_size=(filter_size, filter_size),
                           num_filters=num_filters,
                           step=(conv_step, conv_step),
                           border_mode=border_mode,
                           tied_biases=True,
                           name='conv_{}'.format(i))
             for i, (num_filters, filter_size, conv_step, border_mode)
                 in enumerate(conv_parameters)])

        # Add two trivial channel masks to allow by-channel dropout
        self.convolutions.insert(6, ChannelMask(name='mask_1'))
        self.convolutions.insert(3, ChannelMask(name='mask_0'))

        self.conv_sequence = ConvolutionalSequence(list(interleave([
            self.convolutions,
            (Rectifier() for _ in self.convolutions)
        ])), self.num_channels, self.image_shape)

        # The AllConvNet applies average pooling to combine top-level
        # features across the image.
        self.flattener = GlobalAverageFlattener()

        # Then it inserts one final 10-way FC layer before softmax
        # self.top_mlp = MLP([Rectifier(), Softmax()],
        #     [conv_parameters[-1][0], fc_layer, self.output_size])
        self.top_softmax = Softmax()

        application_methods = [
            self.conv_sequence.apply,
            self.flattener.apply,
            self.top_softmax.apply
        ]

        super(AllConvNet, self).__init__(application_methods, **kwargs)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

def create_all_conv_net():
    convnet = AllConvNet(
                    weights_init=HeInitialization(),
                    # weights_init=IsotropicGaussian(0.05),
                    biases_init=Constant(0))
    convnet.initialize()
    return convnet

class NoisyAllConvNet(FeedforwardSequence, Initializable):
    def __init__(self, image_shape=None, output_size=None,
            noise_batch_size=None, noise_after_rectifier=False, **kwargs):
        self.num_channels = 3
        self.image_shape = image_shape or (32, 32)
        self.output_size = output_size or 10
        self.noise_batch_size = noise_batch_size
        conv_parameters = [
                (96, 3, 1, 'half', Convolutional),
                (96, 3, 1, 'half', Convolutional),
                (96, 3, 2, 'half', NoisyConvolutional),
                (192, 3, 1, 'half', Convolutional),
                (192, 3, 1, 'half', Convolutional),
                (192, 3, 2, 'half', NoisyConvolutional),
                (192, 3, 1, 'half', Convolutional),
                (192, 1, 1, 'valid', Convolutional),
                (10, 1, 1, 'valid', Convolutional)
        ]
        fc_layer = 10

        self.convolutions = []
        layers = []
        for i, (num_filters, filter_size, conv_step, border_mode, cls
                ) in enumerate(conv_parameters):
            if cls == NoisyConvolutional and noise_after_rectifier:
                cls = NoisyConvolutional2
            layer = cls(filter_size=(filter_size, filter_size),
                           num_filters=num_filters,
                           step=(conv_step, conv_step),
                           border_mode=border_mode,
                           tied_biases=True,
                           name='conv_{}'.format(i))
            if cls == NoisyConvolutional or cls == NoisyConvolutional2:
                layer.noise_batch_size = self.noise_batch_size
            self.convolutions.append(layer)
            layers.append(layer)
            if cls != NoisyConvolutional2:
                layers.append(Rectifier())

        self.conv_sequence = ConvolutionalSequence(layers,
                self.num_channels, image_size=self.image_shape)

        # The AllConvNet applies average pooling to combine top-level
        # features across the image.
        self.flattener = GlobalAverageFlattener()

        # Then it inserts one final 10-way FC layer before softmax
        # self.top_mlp = MLP([Rectifier(), Softmax()],
        #     [conv_parameters[-1][0], fc_layer, self.output_size])
        self.top_softmax = Softmax()

        application_methods = [
            self.conv_sequence.apply,
            self.flattener.apply,
            self.top_softmax.apply
        ]

        super(NoisyAllConvNet, self).__init__(application_methods, **kwargs)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

def create_noisy_all_conv_net(noise_batch_size, noise_after_rectifier=False):
    convnet = NoisyAllConvNet(
                    weights_init=HeInitialization(),
                    # weights_init=IsotropicGaussian(0.05),
                    biases_init=Constant(0),
                    noise_batch_size=noise_batch_size,
                    noise_after_rectifier=noise_after_rectifier)
    convnet.initialize()
    return convnet
