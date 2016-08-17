"""Residual network.

He/Zhang/Ren/Sun's "Deep Residual Learning for Image Recognition"
https://arxiv.org/pdf/1512.03385v1.pdf
"""


from blocks.bricks import application, lazy
from blocks.bricks import Brick
from blocks.bricks import FeedforwardSequence
from blocks.bricks import Initializable
from blocks.bricks import MLP
from blocks.bricks import Rectifier
from blocks.bricks import Softmax
from blocks.bricks import SpatialBatchNormalization
from blocks.bricks.conv import Convolutional, ConvolutionalSequence
from blocks.bricks.conv import Flattener, MaxPooling
from blocks.initialization import Constant, Uniform, NdarrayInitialization
from blocks.initialization import IsotropicGaussian
from blocks.utils import repr_attrs
from intent.noisy import SpatialNoise
import numpy
import theano
from theano import tensor
from toolz.itertoolz import interleave


class ResidualConvolutional(Initializable):
    @lazy(allocation=['filter_size', 'num_filters', 'num_channels'])
    def __init__(self, filter_size, num_filters, num_channels,
                 batch_size=None,
                 mid_noise=False,
                 out_noise=False,
                 tied_noise=False,
                 noise_rate=None,
                 noise_batch_size=None,
                 prior_noise_level=None,
                 image_size=(None, None), step=(1, 1),
                 **kwargs):
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.mid_noise = mid_noise
        self.noise_batch_size = noise_batch_size
        self.noise_rate = noise_rate
        self.step = step
        self.border_mode = 'half'
        self.tied_biases = True
        depth = 2

        self.b0 = SpatialBatchNormalization(name='b0')
        self.r0 = Rectifier(name='r0')
        self.n0 = (SpatialNoise(name='n0', noise_rate=self.noise_rate,
                tied_noise=tied_noise,
                prior_noise_level=prior_noise_level) if mid_noise else None)
        self.c0 = Convolutional(name='c0')
        self.b1 = SpatialBatchNormalization(name='b1')
        self.r1 = Rectifier(name='r1')
        self.n1 = (SpatialNoise(name='n1', noise_rate=self.noise_rate,
                tied_noise=tied_noise,
                prior_noise_level=prior_noise_level) if out_noise else None)
        self.c1 = Convolutional(name='c1')
        kwargs.setdefault('children', []).extend([c for c in [
            self.c0, self.b0, self.r0, self.n0,
            self.c1, self.b1, self.r1, self.n1] if c is not None])
        super(ResidualConvolutional, self).__init__(**kwargs)

    def get_dim(self, name):
        if name == 'input_':
            return ((self.num_channels,) + self.image_size)
        if name == 'output':
            return self.c1.get_dim(name)
        return super(ResidualConvolutionalUnit, self).get_dim(name)

    @property
    def num_output_channels(self):
        return self.num_filters

    def _push_allocation_config(self):
        self.b0.input_dim = self.get_dim('input_')
        self.b0.push_allocation_config()
        if self.r0:
            self.r0.push_allocation_config()
        if self.n0:
            self.n0.noise_batch_size = self.noise_batch_size
            self.n0.num_channels = self.num_channels
            self.n0.image_size = self.image_size
        self.c0.filter_size = self.filter_size
        self.c0.batch_size = self.batch_size
        self.c0.num_channels = self.num_channels
        self.c0.num_filters = self.num_filters
        self.c0.border_mode = self.border_mode
        self.c0.image_size = self.image_size
        self.c0.step = self.step
        self.c0.use_bias = False
        self.c0.push_allocation_config()
        c0_shape = self.c0.get_dim('output')
        self.b1.input_dim = c0_shape
        self.b1.push_allocation_config()
        self.r1.push_allocation_config()
        if self.n1:
            self.n1.noise_batch_size = self.noise_batch_size
            self.n1.num_channels = self.num_filters
            self.n1.image_size = c0_shape[1:]
        self.c1.filter_size = self.filter_size
        self.c1.batch_size = self.batch_size
        self.c1.num_channels = self.num_filters
        self.c1.num_filters = self.num_filters
        self.c1.border_mode = self.border_mode
        self.c1.image_size = c0_shape[1:]
        self.c1.step = (1, 1)
        self.c1.use_bias = False
        self.c1.push_allocation_config()

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        shortcut = input_
        # Batchnorm, then Relu, then Convolution
        first_conv = self.b0.apply(input_)
        first_conv = self.r0.apply(first_conv)
        if self.n0:
            first_conv = self.n0.apply(first_conv)
        first_conv = self.c0.apply(first_conv)
        # Batchnorm, then Relu, then Convolution (second time)
        second_conv = self.b1.apply(first_conv)
        second_conv = self.r1.apply(second_conv)
        if self.n1:
            second_conv = self.n1.apply(second_conv)
        residual = second_conv

        # Apply stride and zero-padding to match shortcut to output
        if self.step and self.step != (1, 1):
            shortcut = shortcut[:,:,::self.step[0],::self.step[1]]
        if self.num_filters > self.num_channels:
            padshape = (residual.shape[0],
                    self.num_filters - self.num_channels,
                    residual.shape[2], residual.shape[3])
            shortcut = tensor.concatenate(
                    [shortcut, tensor.zeros(padshape, dtype=residual.dtype)],
                    axis=1)
        elif self.num_filters < self.num_channels:
            shortcut = shortcut[:,:self.num_channels,:,:]

        response = shortcut + residual
        return response

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

class ResNet(FeedforwardSequence, Initializable):
    def __init__(self, image_size=None, output_size=None,
            mid_noise=False, out_noise=False, tied_noise=False,
            noise_batch_size=None,
            noise_rate=None, prior_noise_level=None,
            **kwargs):
        self.num_channels = 3
        self.image_size = image_size or (32, 32)
        self.output_size = output_size or 10
        self.noise_batch_size = noise_batch_size
        self.noise_rate = noise_rate
        n = 16
        num_filters = [16, 32, 64]
        num_channels = num_filters[0]
        self.convolutions = [
            Convolutional(
                    filter_size=(3, 3),
                    num_filters=num_channels,
                    step=(1, 1),
                    border_mode='half',
                    tied_biases=True,
                    name='conv_0'),
        ]
        for j, num in enumerate(num_filters):
            for i in range(n):
                self.convolutions.append(ResidualConvolutional(
                    filter_size=(3, 3),
                    num_filters=num,
                    num_channels=num_channels,
                    mid_noise=mid_noise,
                    out_noise=out_noise,
                    tied_noise=tied_noise,
                    noise_rate=noise_rate,
                    noise_batch_size=noise_batch_size,
                    prior_noise_level=prior_noise_level,
                    step=(2, 2) if i == 0 and j > 0 else (1, 1),
                    name='group_%d_%d' % (num, i)
                ))
                num_channels = num
        self.convolutions.extend([
            SpatialBatchNormalization(name='bn_last'),
            Rectifier(name='relu_last')
       ])
        self.conv_sequence = ConvolutionalSequence(
                self.convolutions,
                image_size=self.image_size,
                num_channels=self.num_channels)
        
        # The AllConvNet applies average pooling to combine top-level
        # features across the image. 
        self.flattener = GlobalAverageFlattener()

        # Then it inserts one final 10-way FC layer before softmax
        self.top_mlp = MLP([Softmax()],
              [num_filters[-1], self.output_size])
        # self.top_softmax = Softmax()

        application_methods = [
            self.conv_sequence.apply,
            self.flattener.apply,
            self.top_mlp.apply
        ]

        super(ResNet, self).__init__(application_methods, **kwargs)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

def create_res_net(mid_noise=False, out_noise=False, tied_noise=False,
         noise_batch_size=None, noise_rate=None, prior_noise_level=None):
    net = ResNet(
        mid_noise=mid_noise,
        out_noise=out_noise,
        tied_noise=tied_noise,
        noise_batch_size=noise_batch_size,
        noise_rate=noise_rate,
        prior_noise_level=prior_noise_level,
        weights_init=HeInitialization(),
        # weights_init=IsotropicGaussian(0.05),
        biases_init=Constant(0))
    net.initialize()
    return net
