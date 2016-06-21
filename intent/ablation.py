from blocks.monitoring.aggregation import AggregationScheme
from blocks.monitoring.aggregation import Aggregator
from blocks.bricks.base import application
from blocks.bricks.base import Brick
from blocks.utils import shared_like
from theano.ifelse import ifelse
from theano import tensor
import numpy

class Sum(AggregationScheme):
    """Aggregation scheme which computes the sum.
    Parameters
    ----------
    variable : :class:`~tensor.TensorVariable`
        Theano variable for the quantity to be summed
    """
    def __init__(self, variable):
        self.variable = variable

    def get_aggregator(self):
        initialized = shared_like(0.)
        total_acc = shared_like(self.variable)

        total_zeros = tensor.as_tensor(self.variable).zeros_like()

        conditional_update_num = self.variable + ifelse(initialized,
                                                         total_acc,
                                                         total_zeros)

        initialization_updates = [(total_acc,
                                   tensor.zeros_like(total_acc)),
                                  (initialized,
                                   tensor.zeros_like(initialized))]

        accumulation_updates = [(total_acc,
                                 conditional_update_num),
                                (initialized, tensor.ones_like(initialized))]

        aggregator = Aggregator(aggregation_scheme=self,
                                initialization_updates=initialization_updates,
                                accumulation_updates=accumulation_updates,
                                readout_variable=(total_acc))

        return aggregator

class ConfusionMatrix(Brick):
    """Confusion Matrix.

    (correct_labels, predicted_labels)

    Outputs a matrix with rates for each combination of
    correct and predicted labels.
    """
    @application(outputs=["confusion_matrix"])
    def apply(self, y, y_hat):
        predicted = y_hat.argmax(axis=1)
        expanded_y = tensor.extra_ops.to_one_hot(y, y_hat.shape[1])
        expanded_y_hat = tensor.extra_ops.to_one_hot(
                predicted, y_hat.shape[1])
        counts = tensor.tensordot(expanded_y, expanded_y_hat, axes=[[0], [0]])
        return counts.astype('int32')

class ConfusionImage(Brick):
    """Confusion Image.

    (correct_labels, predicted_labels)

    Outputs a matrix with sum of images in each cell of the confusion matrix
    table.
    """
    @application(outputs=["confusion_image"])
    def apply(self, y, y_hat, x):
        predicted = y_hat.argmax(axis=1)
        expanded = numpy.zeros(
                (y.shape[0], y_hat.shape[1], y_hat.shape[1]) + x.shape)
        expanded[(numpy.range[y.shape[0]], y, predicted) +
                (slice(None),) * len(x.shape - 1)] = x
        result = expanded.sum(axis=0)
        return result

def ablate_inputs(ablation, activations, weights, axis=None, compensate=True):
    """
    Zeros incoming weights for the given set of input neurons, and
    then approximates the response of those neurons by adjusting
    weights of other inputs, using lienar least squares approximation
    based on activation data of the input neurons.

    ablation - a list of indexes of inputs to zero.  Each index should
        be 0 <= index < len(input_neurons).
    activations - a (samples, input_neurons) matrix of activation data
        for the input neurons.
    weights - a (output_neurons, input_neurons, *...) matrix of weights
        for the next layer of neurons.  The input neuron axis is
        specified by axis, defaulted to 1 (the second axis).
    axis - the axis over which input neurons are listed in the
        weight matrix.
    """
    if len(weights.shape) <= 2:
        axis = 0  # Assume a linear weight, where output is on axis 0
    else:
        axis = 1  # Assume a convolutional weight, where output is on axis 1
    # B contains the activations of the removed neurons
    B = activations[:, ablation]
    # A contains the activations of the non-removed neurons
    remaining = numpy.ones(activations.shape[1], numpy.bool)
    remaining[ablation] = 0
    A = activations[:, remaining]
    # Solve x = np.linagl.lstsq(A, B) for the original layer
    # X dimensions: (remaining, removed)
    (X, res, rank, s) = numpy.linalg.lstsq(A, B)
    # Extract the set of weights that have been removed
    # import pdb; pdb.set_trace()
    removed_weights = numpy.take(weights, ablation, axis=axis)
    # Compute weight adjustment that approximates the removed neurons.
    adj_weights = (numpy.tensordot(X, removed_weights, axes=([1], [axis]))
            .swapaxes(0, axis))
    # Construct the result
    result = numpy.copy(weights)
    dims = range(len(result.shape))
    # Zero direct inputs from the ablated neurons
    result[tuple(ablation
        if a == axis else slice(None) for a in dims)] = 0
    # Adjust inputs from all the other neurons
    if compensate:
        result[tuple(remaining
            if a == axis else slice(None) for a in dims)] += adj_weights
    return result
