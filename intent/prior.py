from theano import tensor

from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from intent.rf import center_location
from intent.rf import layerarray_fieldmap
from theano import tensor
from blocks.roles import OUTPUT
import theano
import numpy

def create_fair_basis(dataset, num_classes, examples_per_class):
    state = dataset.open()
    feature, target = dataset.get_data(state=state, request=[0])

    basis = numpy.zeros((num_classes * examples_per_class,) +
                    feature.shape[1:], dtype=theano.config.floatX)
    counters = [0] * num_classes
    index = 0
    while min(counters) < examples_per_class:
        feature, target = dataset.get_data(state=state, request=[index])
        target = target[0, 0]
        feature = feature / 256
        if target < num_classes and counters[target] < examples_per_class:
            basis[target + counters[target] * num_classes, :, :, :] = (
                            feature[0, :, :, :])
            counters[target] += 1
        index += 1
    dataset.close(state=state)

    return basis

def _cropped_slices(offset, size):
    corner = 0
    if offset < 0:
        corner = -offset
        size += offset
        offset = 0
    elif offset > 0:
        size -= offset

    return (slice(corner, corner + size), slice(offset, offset + size))

def _center_slices(center, shape):
    # offset = tuple(s // 2 - c for s, c in zip(shape, center))
    # (xto, xfrom), (yto, yfrom) = (
    #         _cropped_slices(o, s) for o, s in zip(offset, shape))
    #(xto, xfrom), (yto, yfrom) = (
    (xto, xfrom), (yto, yfrom) = (
            _cropped_slices(s // 2 - c, s) for c, s in zip(center, shape))
    return (xto, yto), (xfrom, yfrom)

def make_shifted_basis(basis, convnet, layers):
    x = tensor.tensor4('features')
    probs = convnet.apply(x)
    cg = ComputationGraph([probs])
    outputs = VariableFilter(roles=[OUTPUT], bricks=layers)(cg.variables)
    fn = theano.function([x], outputs)
    results = fn(basis)
    shifted = []
    for result, layer in zip(results, layers):
        fieldmap = layerarray_fieldmap(
                        convnet.layers[0:convnet.layers.index(layer) + 1])
        # result is 4d (basis_case, unit, dimx, dimy)
        flat_result = result.reshape(result.shape[:2] + (-1, ))
        fargmax = flat_result.argmax(axis=2)
        act_locations = numpy.stack(
                divmod(fargmax, result.shape[2]), axis=-1)
        print(layer.name, 'shape is', result.shape, 'and fieldmap is', fieldmap)
        print('max act locations', act_locations)
        # imlocations is 3d (basis_case, unit, 2=[x,y])
        im_locations = center_location(fieldmap, act_locations)
        print('image locations', im_locations)
        # basis is 4d (basis_case, 1, dimx, dimy)
        # newbasis will be 5d (unit, basis_case, 1, dimx, dimy)
        newbasis = numpy.zeros((result.shape[1],) + basis.shape,
                dtype=theano.config.floatX)
        for b in range(result.shape[0]):
            for u in range(result.shape[1]):
                toslices, fromslices = _center_slices(
                        im_locations[b, u], basis.shape[2:])
                newbasis[(u, b, 0) + toslices] = basis[(b, 0) + fromslices]
        shifted.append(newbasis)
    return tuple(shifted)

