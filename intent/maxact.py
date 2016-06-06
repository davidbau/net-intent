import logging
import numpy as np
from collections import OrderedDict, defaultdict

from theano import tensor
from theano import gradient
from theano.sandbox import cuda

from blocks.algorithms import UpdatesAlgorithm
from blocks.bricks import application
from blocks.bricks import Linear
from blocks.bricks.base import Brick
from blocks.filter import get_brick
from blocks.roles import PersistentRole
from blocks.roles import add_role
from blocks.utils import shared_floatx_zeros
from picklable_itertools.extras import equizip
import theano
import numpy
import numbers
from theano.printing import Print

class MaximumActivationStatisticsRole(PersistentRole):
    pass

class MaximumActivationQuantityRole(MaximumActivationStatisticsRole):
    pass

class MaximumActivationIndexRole(MaximumActivationStatisticsRole):
    pass

# role for attribution historgram
MAXIMUM_ACTIVATION_QUANTITY = MaximumActivationQuantityRole()
MAXIMUM_ACTIVATION_INDEX = MaximumActivationIndexRole()

def _axis_count(shape, axis, ndim):
    return tensor.arange(shape[axis]).dimshuffle(
            tuple(0 if axis == a else 'x' for a in range(ndim)))

def _apply_perm(data, indices, axis=0):
    """
    Does smart indexing of data according to indices along an axis.

    Indicies is a tensor of indices shaped like data except along
    the given axis, which can be of any length.  The indicies select
    locations along that axis.

    _apply_perm can be used to apply the tensor permutation returned
    from tensor.argsort().
    """
    ndim = data.type.ndim
    shape = data.shape
    slicedataay = []
    return data[tuple(indices if a == axis else
            _axis_count(shape, a, ndim) for a in range(ndim))]

def _apply_index(data, indices, axis=0):
    """
    Indexes data along a single axis.

    Indicies is a tensor of indices shaped like data minus the given axis.
    The result is a slice of data removing the given axis: for each entry
    of the othe dimiensions, the given index for that axis is used to select
    the single item.

    _apply_index can be used to derefernce the tensor search results
    returned from tensor.argmax().
    """
    ndim = data.type.ndim
    shape = data.shape
    slicedataay = []
    return data[tuple(indices if a == axis else
            _axis_count(shape, a, ndim - 1) if a < axis else
            _axis_count(shape, a - 1, ndim - 1)
            for a in range(ndim))]

def _create_maximum_activation_for(output, topn, dims=None):
    # Automatically compute the number of units
    if dims is None:
        dims = get_brick(output).get_dims(['output'])[0]
    if isinstance(dims, numbers.Integral):
        dims = (dims,)
        index = theano.shared(numpy.zeros((topn, dims[0]), dtype=numpy.int))
    else:
        index = theano.shared(numpy.zeros((topn, dims[0], 2), dtype=numpy.int))

    quantity = shared_floatx_zeros((topn, dims[0]))

    index.tag.for_output = output
    add_role(index, MAXIMUM_ACTIVATION_INDEX)
    quantity.tag.for_output = output
    add_role(quantity, MAXIMUM_ACTIVATION_QUANTITY)

    return (dims, quantity, index)

def _create_maximum_activation_update(output, record, streamindex, topn):
    """
    Calculates update of the topn maximums for one batch of outputs.
    """
    dims, maximums, indices = record
    counters = tensor.tile(tensor.shape_padright(
        tensor.arange(output.shape[0]) + streamindex), (1, output.shape[1]))
    if len(dims) == 1:
        # output is a 2d tensor, (cases, units) -> activation
        tmax = output
        # counters is a 2d tensor broadcastable (cases, units) -> case_index
        tind = counters
    else:
        # output is a 4d tensor: fmax flattens it to 3d
        fmax = output.flatten(ndim=3)
        # fargmax is a 2d tensor containing maximum locations for each output
        fargmax = fmax.argmax(axis=2)
        # now take the maximum. tmax is 2d, (cases, units) -> activation
        tmax = _apply_index(fmax, fargmax, axis=2)
        # tind is a 3d tensor (cases, units, 2) -> case_index, maxloc_index
        # this will match indices which is a 3d tensor also
        tind = tensor.stack((counters, fargmax), axis=2)
    cmax = tensor.concatenate((maximums, tmax), axis=0)
    cind = tensor.concatenate((indices, tind), axis=0)
    cargsort = cmax.argsort(axis=0)[:topn]
    newmax = _apply_perm(cmax, cargsort, axis=0)
    if cind.type.ndim > 2:
        cargsort = tensor.tile(tensor.shape_padright(cargsort),
                (1, 1, cind.shape[2]))
    newind = _apply_perm(cind, cargsort, axis=0)
    return [(maximums, newmax), (indices, newind)]

class MaximumActivationSearch(UpdatesAlgorithm):
    """Algorithm for identifying maximum activations for individual neurons.

    Parameters
    ----------
    outputs: list of variables that should be probed.
    """

    def __init__(self, outputs=None, dims=None, topn=100, **kwargs):
        self.outputs = outputs
        self.topn = topn
        self.dims = dims if dims is not None else {}
        # Represents the location within the datastream
        self.streamindex = theano.shared(numpy.zeros((), dtype=numpy.int64))
        super(MaximumActivationSearch, self).__init__(**kwargs)

        self.maximum_activations = OrderedDict(
            [(o, _create_maximum_activation_for(o, self.topn,
                dims=self.dims.get(o, None))) for o in self.outputs])
        self.maximum_activations_updates = []
        for o, record in self.maximum_activations.items():
            self.maximum_activations_updates.extend(
                _create_maximum_activation_update(
                    o, record, self.streamindex, self.topn))
        self.maximum_activations_updates.append(
            (self.streamindex, self.streamindex + self.outputs[0].shape[0]))
        self.add_updates(self.maximum_activations_updates)
