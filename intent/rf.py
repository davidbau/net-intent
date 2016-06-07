from blocks.bricks.conv import Convolutional
from blocks.bricks.conv import Pooling
from blocks.bricks.conv import ConvolutionalSequence
from blocks.bricks.conv import Flattener
from scipy.ndimage.filters import gaussian_filter
import numpy

def receptive_field(location, fieldmap):
    """Computes the receptive field of a specific location.

    Parameters
    ----------
    location: tuple
        The x-y position of the unit being queried.
    fieldmap:
        The (offset, size, step) tuple fieldmap representing the
        receptive field map for the layer being queried.
    """
    return compose_fieldmap(rf, (location, (1, 1), (1, 1)))[:2]

def layer_fieldmap(brick):
    if isinstance(brick, Convolutional):
        size = brick.filter_size
        offset = edge_mode_offset(brick.border_mode or 'valid', size)
        step = brick.step or (1, 1)
        return (offset, size, step)
    if isinstance(brick, Pooling):
        size = brick.pooling_size
        offset = edge_mode_offset(brick.padding or (0, 0), size)
        step = brick.step or size
        return (offset, size, step)
    if isinstance(brick, ConvolutionalSequence):
        return layerarray_fieldmap(brick.layers)
    if isinstance(brick, Flattener):
        return ((0, 0), (float('inf'), float('inf')), (1, 1))
    return ((0, 0), (1, 1), (1, 1))

def layerarray_fieldmap(layerarray):
    fieldmap = ((0, 0), (1, 1), (1, 1))
    for layer in layerarray:
        fieldmap = compose_fieldmap(fieldmap, layer_fieldmap(layer))
    return fieldmap

def is_iterable(x):
    return hasattr(x, '__iter__')

def edge_mode_offset(mode, size):
    if not is_iterable(size):
       size = (size, size)
    if mode == 'full':
       return tuple(1 - s for s in size)
    if mode == 'valid':
       return tuple(0 for s in size)
    if mode == 'same' or mode == 'half':
       return tuple(-(s // 2) for s in size)
    return tuple(-edge for edge in mode)[:2]

# rf1 is the lower layer, rf2 is the higher layer
def compose_fieldmap(rf1, rf2):
    """Composes two stacked fieldmap maps.

    Field maps are represented as triples of (offset, size, step),
    where each is an (x, y) pair.

    To find the pixel range corresponding to output pixel (x, y), just
    do the following:
       start_x = x * step[0] + offset[1]
       start_y = y * step[1] + offset[1]

    Parameters
    ----------
    rf1: tuple
        The lower-layer receptive fieldmap, a tuple of (offset, size, step).
    rf2: tuple
        The higher-layer receptive fieldmap, a tuple of (offset, size, step).
    """
    offset1, size1, step1 = rf1
    offset2, size2, step2 = rf2

    size = tuple((size2c - 1) * step1c + size1c
            for size1c, step1c, size2c in zip(size1, step1, size2))
    offset = tuple(offset2c * step1c + offset1c
            for offset2c, step1c, offset1c in zip(offset2, step1, offset1))
    step = tuple(step2c * step1c
            for step1c, step2c in zip(step1, step2))
    return (offset, size, step)

def _cropped_slices(offset, size, limit):
    corner = 0
    if offset < 0:
        size += offset
        offset = 0
    if limit - offset < size:
        corner = limit - offset
        size -= corner
    return (slice(corner, corner + size), slice(offset, offset + size))

def crop_field(image_data, fieldmap, location):
    """Crops image_data to the specified receptive field.

    Together fieldmap and location specify a receptive field on the image,
    which may overlap the edge. This returns a crop to that shape, including
    any zero padding necessary to fill out the shape beyond the image edge.
    """
    coloraxis = 0 if image_data.size <= 2 else 1
    allcolors = () if not coloraxis else (slice(None),) * coloraxis
    colordepth = () if not coloraxis else (image_data.size[0], )
    offset, size = receptive_field(fieldmap, location)
    result = numpy.zeros(colordepth + size)
    (xto, xfrom), (yto, yfrom) = (_cropped_slices(
        o, s, l) for o, s, l in zip(offset, size, image_data.size[coloraxis:]))
    result[allcolors + (xto, yto)] = image_data[allcolors + (xfrom, yfrom)]
    return result

def _gaussian_1d(size, sigma):
    n = numpy.arange(0, size) - (size - 1.0) / 2.0
    sig2 = 2 * sigma * sigma
    w = numpy.exp(-n ** 2 / sig2)
    return w

def _gaussian(shape, sigmafrac=0.2):
    return (_gaussian_1d(shape[0], sigmafrac * shape[0])[:, numpy.newaxis] *
            _gaussian_1d(shape[1], sigmafrac * shape[1])[numpy.newaxis, :])

def _centered_slice(fieldmap, activation_shape):
    offset, size, step = fieldmap
    return tuple(slice(s // 2 + o, s // 2 + o + a * t, t)
            for o, s, t, a in zip(offset, size, step, activation_shape))

def make_mask(image_shape, fieldmap, activation_data,
                sigma=0.2, threshold=0.9, alpha=0.1):
    """Creates a receptive field mask as described by Khosla.

    The activation data is inverted through the fieldmap and then
    convolved with a gaussian; then finally the convolution is clipped.
    """
    offset, shape, step = fieldmap
    activations = numpy.zeros(image_shape)
    activations[_centered_slice(fieldmap, activation_data.shape)] = (
        activation_data)
    blurred = gaussian_filter(
        activations, sigma=tuple(s * sigma for s in shape), mode='constant')
    maximum = blurred.flatten().max()
    return 1 - (1 - alpha) * (blurred < maximum * 0.9)
