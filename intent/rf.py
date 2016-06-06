from blocks.bricks import Convolutional, Pooling, ConvolutionalSequence
from blocks.bricks import ConvolutionalSequence
from blocks.bricks import Flattener

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
    return compose_field(fieldmap, as_fieldmap(location))[:2]

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

def as_fieldmap(location):
    location = tuple(location)
    if len(location) == 2 and not is_iterable(location[0]):
       location = (location,)
    if len(location) == 1:
       location = location + ((1, 1),)
    if len(location) == 2:
       location = location + ((1, 1),)
    return location

def edge_mode_offset(mode, size):
    if not is_iterable(size):
       size = (size, size)
    if mode == 'full'
       return tuple(1 - s for s in size)
    if mode == 'valid'
       return tuple(0 for s in size)
    if mode == 'same' or mode == 'helf':
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

