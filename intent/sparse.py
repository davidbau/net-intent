from blocks.algorithms import GradientDescent
import theano
import numpy

class SparseGradientDescent(GradientDescent):
    def __init__(self, mask=None, **kwargs):
        super(SparseGradientDescent, self).__init__(**kwargs)
        # Apply masks to updates of specified parameters
        self.updates = [
                (param, mask[param] * update) if param in mask else (param, update)
                for param, update in self.updates]

def fc_mask(output_size):
    # Create a sparse mask for updates
    mask = numpy.zeros(((output_size - 1) * output_size, output_size),
            dtype=theano.config.floatX)
    output_size = mask.shape[1]
    for i in range(output_size):
        for j in range(output_size - 1):
            mask[i * (output_size - 1) + j, i] = 1
            mask[j * (output_size - 1) + i - 1 + (output_size if j >= i else 0), i] = 1
    return mask
