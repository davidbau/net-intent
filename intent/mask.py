import logging
import collections
import numpy as np
import theano
from theano import tensor
from blocks.bricks import Brick
from blocks.bricks import Feedforward
from blocks.bricks import FeedforwardSequence
from blocks.bricks import Initializable
from blocks.bricks import application
from blocks.bricks import lazy
from blocks.roles import add_role, AuxiliaryRole, ParameterRole, VariableRole


class ChannelMaskRole(VariableRole):
    pass

# Role for parameters that are used to inject noise during training.
CHANNEL_MASK = ChannelMaskRole()

class ChannelMask(Feedforward):
    """Allows dropout to apply to an entire channel at once over the entire image
    as opposed to dropping out different channels for each pixel.
    """
    @lazy(allocation=['input_dim'])
    def __init__(self, input_dim, **kwargs):
        self.input_dim = input_dim
        super(ChannelMask, self).__init__(**kwargs)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        # Does this need to be a shared var?
        mask = tensor.ones(input_.shape[:2])
        add_role(mask, CHANNEL_MASK)
        return input_ * tensor.shape_padright(mask, n_ones=2)

    # Needed for the Feedforward interface.
    @property
    def output_dim(self):
        return self.input_dim

    # The following properties allow for BatchNormalization bricks
    # to be used directly inside of a ConvolutionalSequence.
    @property
    def image_size(self):
        return self.input_dim[-2:]

    @image_size.setter
    def image_size(self, value):
        if not isinstance(self.input_dim, collections.Sequence):
            self.input_dim = (None,) + tuple(value)
        else:
            self.input_dim = (self.input_dim[0],) + tuple(value)

    @property
    def num_channels(self):
        return self.input_dim[0]

    @num_channels.setter
    def num_channels(self, value):
        if not isinstance(self.input_dim, collections.Sequence):
            self.input_dim = (value,) + (None, None)
        else:
            self.input_dim = (value,) + self.input_dim[-2:]

    def get_dim(self, name):
        if name in ('input_', 'output'):
            return self.input_dim
        else:
            raise KeyError

    @property
    def num_output_channels(self):
        return self.num_channels

