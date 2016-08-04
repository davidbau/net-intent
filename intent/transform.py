import numpy
from fuel import config

from fuel.transformers.image import ExpectsAxisLabels, SourcewiseTransformer
try:
    from fuel.transformers._image import window_batch_bchw
    window_batch_bchw_available = True
except ImportError:
    window_batch_bchw_available = False


class RandomFlip(SourcewiseTransformer, ExpectsAxisLabels):
    """Randomly flip images left-right
    Parameters
    ----------
    data_stream : :class:`AbstractDataStream`
        The data stream to wrap.
    """
    def __init__(self, data_stream, **kwargs):
        self.rng = kwargs.pop('rng', None)
        self.warned_axis_labels = False
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(RandomFlip, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, source, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        if isinstance(source, numpy.ndarray) and source.ndim == 4:
            batch_size = source.shape[0]
            whichflip = self.rng.binomial(1, 0.5, batch_size)
            # TODO: determine if we are allowed to flip in-place
            source[whichflip,:,:,:] = source[whichflip,:,:,::-1]
            return source
        elif all(isinstance(b, numpy.ndarray) and b.ndim == 3 for b in source):
            return [self.transform_source_example(im, source_name)
                    for im in source]
        else:
            raise ValueError("uninterpretable batch format; expected a list "
                             "of arrays with ndim = 3, or an array with "
                             "ndim = 4")

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        if not isinstance(example, numpy.ndarray) or example.ndim != 3:
            raise ValueError("uninterpretable example format; expected "
                             "ndarray with ndim = 3")
        flip = self.rng.binomial(1, 0.5)
        if flip:
            return example[:,:,::-1]
        else:
            return example

class NormalizeBatchLevels(SourcewiseTransformer, ExpectsAxisLabels):

    def __init__(self, data_stream, **kwargs):
        self.warned_axis_labels = False
        kwargs.setdefault('produces_examples', False)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(NormalizeBatchLevels, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, source, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        if isinstance(source, numpy.ndarray) and source.ndim == 4:
            mean_levels = source.mean(axis=(0, 2, 3), keepdims=True)
            zeroed = source - mean_levels
            std = zeroed.std(axis=(0, 2, 3), keepdims=True)
            return zeroed / std
        else:
            raise ValueError("uninterpretable batch format; expected an "
                             "array with ndim = 4")

class RandomPadCropFlip(SourcewiseTransformer, ExpectsAxisLabels):
    """Randomly crop images to a fixed window size.
    Parameters
    ----------
    data_stream : :class:`AbstractDataStream`
        The data stream to wrap.
    window_shape : tuple
        The `(height, width)` tuple representing the size of the output
        window.
    Notes
    -----
    This transformer expects to act on stream sources which provide one of
     * Single images represented as 3-dimensional ndarrays, with layout
       `(channel, height, width)`.
     * Batches of images represented as lists of 3-dimensional ndarrays,
       possibly of different shapes (i.e. images of differing
       heights/widths).
     * Batches of images represented as 4-dimensional ndarrays, with
       layout `(batch, channel, height, width)`.
    The format of the stream will be un-altered, i.e. if lists are
    yielded by `data_stream` then lists will be yielded by this
    transformer.
    """
    def __init__(self, data_stream, window_shape,
            pad=None, x_flip=True, **kwargs):
        if not window_batch_bchw_available:
            raise ImportError('window_batch_bchw not compiled')
        self.window_shape = window_shape
        self.pad = pad
        self.x_flip = x_flip
        self.rng = kwargs.pop('rng', None)
        self.warned_axis_labels = False
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(RandomPadCropFlip, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, source, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        windowed_height, windowed_width = self.window_shape
        if isinstance(source, numpy.ndarray) and source.ndim == 4:
            # Hardcoded assumption of (batch, channels, height, width).
            # This is what the fast Cython code supports.
            out = numpy.empty(source.shape[:2] + self.window_shape,
                              dtype=source.dtype)
            batch_size = source.shape[0]
            # If padding is requested, pad before making random crop.
            if self.pad is not None:
                symmetric = (self.pad, self.pad)
                paddims = ((0, 0), (0, 0), symmetric, symmetric)
                source = numpy.pad(source, paddims, 'constant')
            image_height, image_width = source.shape[2:]
            max_h_off = image_height - windowed_height
            max_w_off = image_width - windowed_width
            if max_h_off < 0 or max_w_off < 0:
                raise ValueError("Got ndarray batch with image dimensions {} "
                                 "but requested window shape of {}".format(
                                     source.shape[2:], self.window_shape))
            offsets_w = self.rng.random_integers(0, max_w_off, size=batch_size)
            offsets_h = self.rng.random_integers(0, max_h_off, size=batch_size)
            window_batch_bchw(source, offsets_h, offsets_w, out)
            # If flipping is requested, randomly flip images horizontally
            if self.x_flip:
                whichflip = self.rng.binomial(1, 0.5, batch_size)
                out[whichflip,:,:,:] = out[whichflip,:,:,::-1]
            return out
        elif all(isinstance(b, numpy.ndarray) and b.ndim == 3 for b in source):
            return [self.transform_source_example(im, source_name)
                    for im in source]
        else:
            raise ValueError("uninterpretable batch format; expected a list "
                             "of arrays with ndim = 3, or an array with "
                             "ndim = 4")

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        windowed_height, windowed_width = self.window_shape
        if not isinstance(example, numpy.ndarray) or example.ndim != 3:
            raise ValueError("uninterpretable example format; expected "
                             "ndarray with ndim = 3")
        image_height, image_width = example.shape[1:]
        if image_height < windowed_height or image_width < windowed_width:
            raise ValueError("can't obtain ({}, {}) window from image "
                             "dimensions ({}, {})".format(
                                 windowed_height, windowed_width,
                                 image_height, image_width))
        # Apply padding
        if self.pad is not None:
            symmetric = (self.pad, self.pad)
            paddims = ((0, 0), symmetric, symmetric)
            example = numpy.pad(example, paddims, 'constant')

        # Apply crop
        if image_height - windowed_height > 0:
            off_h = self.rng.random_integers(0, image_height - windowed_height)
        else:
            off_h = 0
        if image_width - windowed_width > 0:
            off_w = self.rng.random_integers(0, image_width - windowed_width)
        else:
            off_w = 0
        example = example[:, off_h:off_h + windowed_height,
                       off_w:off_w + windowed_width]

        # Apply flip
        if self.x_flip and self.rng.binomial(1, 0.5):
            return example[:,:,::-1]
        return example
