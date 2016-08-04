import numpy
from fuel import config

from fuel.transformers.image import ExpectsAxisLabels, SourcewiseTransformer

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
