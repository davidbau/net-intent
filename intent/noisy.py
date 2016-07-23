"""Noisy network elements.
"""
import numpy as np
import theano
from theano import tensor

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
from blocks.initialization import Constant, Uniform
from blocks.roles import VariableRole
from toolz.itertoolz import interleave

class NitsRole(VariableRole):
    pass

# role for synpic historgram
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
    @lazy(allocation=['input_dim', 'output_dim'])
    def __init__(self, input_dim, output_dim,
            prior_mean=0, prior_noise_level=0, **kwargs):
        self.linear = Linear()
        self.mask = Linear(name='mask')
        children = [self.linear, self.mask]
        kwargs.setdefault('children', []).extend(children)
        super(NoisyLinear, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_mean = prior_mean
        self.prior_noise_level = prior_noise_level

    def _push_allocation_config(self):
        self.linear.input_dim = self.input_dim
        self.linear.output_dim = self.output_dim
        self.mask.input_dim = self.input_dim
        self.mask.output_dim = self.output_dim

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
        noise_level = self.mask.apply(input_)
        noise = self.theano_rng.normal(noise_level.shape)
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
    @lazy(allocation=['filter_size', 'num_filters', 'num_channels'])
    def __init__(self, filter_size, num_filters, num_channels, batch_size=None,
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
        self.convolution.batch_size = self.batch_size
        self.convolution.image_size = self.image_size
        self.convolution.step = self.step
        self.convolution.border_mode = self.border_mode
        self.convolution.tied_biases = self.tied_biases
        self.mask.filter_size = self.filter_size
        self.mask.num_filters = self.num_filters
        self.mask.num_channels = self.num_channels
        self.mask.batch_size = self.batch_size
        self.mask.image_size = self.image_size
        self.mask.step = self.step
        self.mask.border_mode = self.border_mode
        self.mask.tied_biases = self.tied_biases

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
        noise_level = self.mask.apply(input_)
        noise = self.theano_rng.normal(noise_level.shape)
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

        self.conv_sequence = ConvolutionalSequence(self.layers, num_channels,
                                                   image_size=image_shape)

        # Construct a top MLP
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims,
                prototype=NoisyLinear())

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


def create_noisy_lenet_5():
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
    convnet = NoisyLeNet(conv_activations, 1, image_size,
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

#    convnet.layers[0].mask.bias_init = (
#            Constant(-1))
#    convnet.layers[3].mask.bias_init = (
#            Constant(-1))
#    convnet.top_mlp.linear_transformations[0].mask.bias_init = (
#            Constant(-1))
#    convnet.top_mlp.linear_transformations[1].mask.bias_init = (
#            Constant(-1))
#    convnet.top_mlp.linear_transformations[2].mask.bias_init = (
#            Constant(-1))

    convnet.initialize()

    return convnet
