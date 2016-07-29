"""Noisy network elements.
"""
import logging
import numpy as np
import theano
from theano import tensor
from theano.printing import Print
from picklable_itertools import chain, repeat, imap
from picklable_itertools.extras import partition_all

from blocks.bricks import Feedforward
from blocks.bricks import FeedforwardSequence
from blocks.bricks import Initializable
from blocks.bricks import Linear
from blocks.bricks import MLP
from blocks.bricks import Random
from blocks.bricks import Rectifier
from blocks.bricks import Softmax
from blocks.bricks import application
from blocks.bricks import lazy
from blocks.bricks.conv import Convolutional, ConvolutionalSequence
from blocks.bricks.conv import Flattener, MaxPooling
from blocks.bricks.interfaces import RNGMixin
from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.initialization import Constant, Uniform, IsotropicGaussian
from blocks.monitoring.evaluators import DatasetEvaluator
from blocks.roles import add_role, AuxiliaryRole, ParameterRole
from blocks.utils import shared_floatx_zeros
from collections import OrderedDict
import fuel
from fuel.schemes import BatchScheme
from toolz.itertoolz import interleave


logger = logging.getLogger(__name__)

class NoiseRole(ParameterRole):
    pass

# Role for parameters that are used to inject noise during training.
NOISE = NoiseRole()


class NitsRole(AuxiliaryRole):
    pass

# Role for variables that quantify the number of nits at a unit.
NITS = NitsRole()

# Annotate all the nits variables
def copy_and_tag_nits(variable, brick):
    """Helper method to copy a variable and annotate it."""
    copy = variable.copy()
    # Theano name
    copy.name = "{}_apply_nits".format(brick.name)
    add_annotation(copy, brick)
    add_annotation(copy, call)
    # Blocks name
    copy.tag.name = name
    add_role(copy, role)
    return copy

class UnitNoiseGenerator(Random):
    def __init__(self, std=1.0, **kwargs):
        self.std = std
        super(UnitNoiseGenerator, self).__init__(**kwargs)

    @application(inputs=['param'], outputs=['output'])
    def apply(self, param):
        return self.theano_rng.normal(param.shape, std=self.std)

class NoiseExtension(SimpleExtension, RNGMixin):
    def __init__(self, noise_parameters=None, **kwargs):
        kwargs.setdefault("before_batch", True)
        self.noise_parameters = noise_parameters
        std = 1.0
        self.noise_init = IsotropicGaussian(std=std)
        self.theano_generator = UnitNoiseGenerator(std=std)
        self.noise_updates = OrderedDict(
            [(param, self.theano_generator.apply(param))
                for param in self.noise_parameters])
        super(NoiseExtension, self).__init__(**kwargs)

    def do(self, callback_name, *args):
        self.parse_args(callback_name, args)
        if callback_name == 'before_training':
            for p in self.parameters:
                self.noise_init.initialize(p, self.rng)
            self.main_loop.algorithm.add_updates(self.noise_updates)

class NoisyLinear(Initializable, Feedforward, Random):
    """Linear transformation sent through a learned noisy channel.

    Parameters
    ----------
    input_dim : int
        The dimension of the input. Required by :meth:`~.Brick.allocate`.
    output_dim : int
        The dimension of the output. Required by :meth:`~.Brick.allocate`.
    num_pieces : int
        The number of linear functions. Required by
        :meth:`~.Brick.allocate`.
    """
    @lazy(allocation=['input_dim', 'output_dim', 'batch_size'])
    def __init__(self, input_dim, output_dim, batch_size,
            prior_mean=0, prior_noise_level=0, **kwargs):
        self.linear = Linear()
        self.mask = Linear(name='mask')
        children = [self.linear, self.mask]
        kwargs.setdefault('children', []).extend(children)
        super(NoisyLinear, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.prior_mean = prior_mean
        self.prior_noise_level = prior_noise_level

    def _push_allocation_config(self):
        self.linear.input_dim = self.input_dim
        self.linear.output_dim = self.output_dim
        self.mask.input_dim = self.output_dim
        self.mask.output_dim = self.output_dim

    def _allocate(self):
        N = shared_floatx_zeros((self.batch_size, self.output_dim), name='N')
        add_role(N, NOISE)
        self.parameters.append(N)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_, application_call):
        """Apply the linear transformation followed by masking with noise.
        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            The input on which to apply the transformations
        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            The transformed input
        """
        pre_noise = self.linear.apply(input_)
        noise_level = -tensor.clip(self.mask.apply(pre_noise), -16, 16)

        # Allow incomplete batches by just taking the noise that is needed
        # noise = Print('noise')(self.parameters[0][:noise_level.shape[0], :])
        noise = self.parameters[0][:noise_level.shape[0], :]
        # noise = Print('noise')(self.theano_rng.normal(noise_level.shape))
        kl = (
            self.prior_noise_level - noise_level 
            + 0.5 * (
                tensor.exp(2 * noise_level)
                + (pre_noise - self.prior_mean) ** 2
                ) / tensor.exp(2 * self.prior_noise_level)
            - 0.5
            )
        application_call.add_auxiliary_variable(kl, roles=[NITS], name='nits')
        return pre_noise + tensor.exp(noise_level) * noise

    def get_dim(self, name):
        if name == 'input_':
            return self.linear.get_dim(name)
        if name == 'output':
            return self.linear.get_dim(name)
        if name == 'nits':
            return self.linear.get_dim('output')
        return super(NoisyLinear, self).get_dim(name)


class NoisyConvolutional(Initializable, Feedforward, Random):
    """Convolutional transformation sent through a learned noisy channel.

    Parameters (same as Convolutional)
    """
    @lazy(allocation=[
        'filter_size', 'num_filters', 'num_channels', 'batch_size'])
    def __init__(self, filter_size, num_filters, num_channels, batch_size,
                 image_size=(None, None), step=(1, 1), border_mode='valid',
                 tied_biases=True,
                 prior_mean=0, prior_noise_level=0, **kwargs):
        self.convolution = Convolutional()
        self.mask = Convolutional(name='mask')
        children = [self.convolution, self.mask]
        kwargs.setdefault('children', []).extend(children)
        super(NoisyConvolutional, self).__init__(**kwargs)
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.image_size = image_size
        self.step = step
        self.border_mode = border_mode
        self.tied_biases = tied_biases
        self.prior_mean = prior_mean
        self.prior_noise_level = prior_noise_level

    def _push_allocation_config(self):
        self.convolution.filter_size = self.filter_size
        self.convolution.num_filters = self.num_filters
        self.convolution.num_channels = self.num_channels
        # self.convolution.batch_size = self.batch_size
        self.convolution.image_size = self.image_size
        self.convolution.step = self.step
        self.convolution.border_mode = self.border_mode
        self.convolution.tied_biases = self.tied_biases
        self.mask.filter_size = (1, 1)
        self.mask.num_filters = self.num_filters
        self.mask.num_channels = self.num_filters
        # self.mask.batch_size = self.batch_size
        self.mask.image_size = self.convolution.get_dim('output')[1:]
        # self.mask.step = self.step
        # self.mask.border_mode = self.border_mode
        self.mask.tied_biases = self.tied_biases

    def _allocate(self):
        out_shape = self.convolution.get_dim('output')
        N = shared_floatx_zeros((self.batch_size,) + out_shape, name='N')
        add_role(N, NOISE)
        self.parameters.append(N)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_, application_call):
        """Apply the linear transformation followed by masking with noise.
        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            The input on which to apply the transformations
        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            The transformed input
        """
        pre_noise = self.convolution.apply(input_)
        # noise_level = self.mask.apply(input_)
        noise_level = tensor.clip(self.mask.apply(pre_noise), -16, 16)
        # Allow incomplete batches by just taking the noise that is needed
        noise = self.parameters[0][:noise_level.shape[0], :, :, :]
        # noise = self.theano_rng.normal(noise_level.shape)
        kl = (
            self.prior_noise_level - noise_level 
            + 0.5 * (
                tensor.exp(2 * noise_level)
                + (pre_noise - self.prior_mean) ** 2
                ) / tensor.exp(2 * self.prior_noise_level)
            - 0.5
            )
        application_call.add_auxiliary_variable(kl, roles=[NITS], name='nits')
        return pre_noise + tensor.exp(noise_level) * noise

    def get_dim(self, name):
        if name == 'input_':
            return self.convolution.get_dim(name)
        if name == 'output':
            return self.convolution.get_dim(name)
        if name == 'nits':
            return self.convolution.get_dim('output')
        return super(NoisyConvolutional, self).get_dim(name)

    @property
    def num_output_channels(self):
        return self.num_filters


class NoisyDataStreamMonitoring(DataStreamMonitoring):
    def __init__(self, variables, data_stream,
            updates=None, noise_parameters=None, **kwargs):
        kwargs.setdefault("after_epoch", True)
        kwargs.setdefault("before_first_epoch", True)
        super(DataStreamMonitoring, self).__init__(**kwargs)
        self._evaluator = DatasetEvaluator(variables, updates)
        self.data_stream = data_stream
        self.noise_parameters = noise_parameters

    def do(self, callback_name, *args):
        """Write the values of monitored variables to the log."""
        logger.info("Monitoring on auxiliary data started")
        saved = [(p, p.get_value()) for p in self.noise_parameters]
        for (p, v) in saved:
            p.set_value(np.zeros(v.shape, dtype=v.dtype))
        value_dict = self._evaluator.evaluate(self.data_stream)
        self.add_records(self.main_loop.log, value_dict.items())
        logger.info("Monitoring on auxiliary data finished")


class NoisyLeNet(FeedforwardSequence, Initializable):
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
    def __init__(self, conv_activations, num_channels, image_shape, batch_size,
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
        self.batch_size = batch_size
        self.top_mlp_activations = top_mlp_activations
        self.top_mlp_dims = top_mlp_dims
        self.border_mode = border_mode
        self.tied_biases = tied_biases

        conv_parameters = zip(filter_sizes, feature_maps)

        # Construct convolutional layers with corresponding parameters
        self.layers = list(interleave([
            (NoisyConvolutional(filter_size=filter_size,
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

        self.conv_sequence = ConvolutionalSequence(
                self.layers, num_channels,
                image_size=image_shape,
                batch_size=self.batch_size)
        self.conv_sequence.name = 'cs'

        # Construct a top MLP
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims,
                prototype=NoisyLinear(batch_size=self.batch_size))

        # We need to flatten the output of the last convolutional layer.
        # This brick accepts a tensor of dimension (batch_size, ...) and
        # returns a matrix (batch_size, features)
        self.flattener = Flattener()
        application_methods = [self.conv_sequence.apply, self.flattener.apply,
                               self.top_mlp.apply]
        super(NoisyLeNet, self).__init__(application_methods, **kwargs)

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


def create_noisy_lenet_5(batch_size):
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
    convnet = NoisyLeNet(conv_activations, 1, image_size, batch_size,
                    filter_sizes=zip(conv_sizes, conv_sizes),
                    feature_maps=feature_maps,
                    pooling_sizes=zip(pool_sizes, pool_sizes),
                    top_mlp_activations=mlp_activations,
                    top_mlp_dims=mlp_hiddens + [output_size],
                    border_mode='valid',
                    weights_init=Constant(0), # Uniform(width=.2),
                    biases_init=Constant(0))

    # We push initialization config to set different initialization schemes
    # for convolutional layers.
    convnet.push_initialization_config()
    convnet.layers[0].convolution.weights_init = (
            Uniform(width=.2))
    convnet.layers[3].convolution.weights_init = (
            Uniform(width=.09))
    convnet.top_mlp.linear_transformations[0].linear.weights_init = (
            Uniform(width=.08))
    convnet.top_mlp.linear_transformations[1].linear.weights_init = (
            Uniform(width=.11))
    convnet.top_mlp.linear_transformations[2].linear.weights_init = (
            Uniform(width=.2))
#
#    convnet.layers[0].mask.weights_init = (
#            Uniform(width=.2))
#    convnet.layers[3].mask.weights_init = (
#            Uniform(width=.09))
#    convnet.top_mlp.linear_transformations[0].mask.weights_init = (
#            Uniform(width=.08))
#    convnet.top_mlp.linear_transformations[1].mask.weights_init = (
#            Uniform(width=.11))
#    convnet.top_mlp.linear_transformations[2].mask.weights_init = (
#            Uniform(width=.2))

#    convnet.layers[0].mask.bias_init = (
#            Constant(8))
#    convnet.layers[3].mask.bias_init = (
#            Constant(8))
#    convnet.top_mlp.linear_transformations[0].mask.bias_init = (
#            Constant(8))
#    convnet.top_mlp.linear_transformations[1].mask.bias_init = (
#            Constant(8))
#    convnet.top_mlp.linear_transformations[2].mask.bias_init = (
#            Constant(8))

    convnet.initialize()

    return convnet

class SampledScheme(BatchScheme):
    """Sampled batches iterator.
    Like shuffledScheme but uses a sampling method instead, and makes
    the final batch complete.
    """
    def __init__(self, *args, **kwargs):
        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = np.random.RandomState(fuel.config.default_seed)
        super(SampledScheme, self).__init__(*args, **kwargs)

    def get_request_iterator(self):
        indices = list(self.indices)
        count = len(indices)
        if count % self.batch_size:
            count += self.batch_size - self.batch_size % count
        self.rng.choice(indices, count)
        return imap(list, partition_all(self.batch_size, indices))
