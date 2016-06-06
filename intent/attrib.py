import logging
import numpy as np
from collections import OrderedDict, defaultdict

from theano import tensor
from theano import gradient

from blocks.algorithms import GradientDescent
from blocks.bricks import application
from blocks.bricks import Linear
from blocks.bricks.base import Brick
from blocks.roles import PersistentRole
from blocks.roles import add_role
from blocks.utils import shared_floatx_zeros
from picklable_itertools.extras import equizip

class ComponentwiseCrossEntropy(Brick):
    """By-component cross entropy.

    Outputs a vector with cross entropy separated out to components,
    one for each label.  If normal cross entropy is -Sum[i](y_i log(y_hat_i)),
    then componentwise cross entropy sepearates this into one component
    per component of y: c_i = -y_i log(y_hat_i).  For categorical input
    data, this vector is zero in all spots except for the labeled category,
    at which point it takes the value -log(y_hat_i).
    """
    @application(outputs=["components"])
    def apply(self, y, y_hat):
        # outputs a vector with cross entropy for each row
        total = tensor.nnet.crossentropy_categorical_1hot(y_hat, y)
        # now place the cross entropy in the column for that class
        splitout = (
            total.dimshuffle(0, 'x') *
            tensor.extra_ops.to_one_hot(y, y_hat.shape[1]))
        # now average the cross entropy componentwise
        components = splitout.mean(axis=0)
        return components

class AttributionStatisticsRole(PersistentRole):
    pass

# role for attribution historgram
ATTRIBUTION_STATISTICS = AttributionStatisticsRole()

def _create_attribution_histogram_for(param, components_size):
    buf = shared_floatx_zeros((components_size,) + param.get_value().shape)
    # buf = tensor.TensorType('floatX', (False,) * (param.ndim + 1))()
    buf.tag.for_parameter = param
    add_role(buf, ATTRIBUTION_STATISTICS)
    return buf

def _create_attribution_updates(attribution, jacobian):
    return (attribution, attribution + jacobian)

class AttributedGradientDescent(GradientDescent):
    def __init__(self, components=None, components_size=None,
                 jacobians=None, **kwargs):
        super(AttributedGradientDescent, self).__init__(**kwargs)
        self.components = components
        self.components_size = components_size
        self.jacobians = jacobians
        if not self.jacobians:
            self.jacobians = self._compute_jacobians()
        self.attributions = OrderedDict(
            [(param, _create_attribution_histogram_for(param, components_size))
                for param in self.parameters])
        self.attribution_updates = OrderedDict(
            [_create_attribution_updates(self.attributions[param],
                self.jacobians[param]) for param in self.parameters])
        self.add_updates(self.attribution_updates)

    def _compute_jacobians(self):
        if self.components is None or self.components.ndim == 0:
            raise ValueError("can't infer jacobians; no components specified")
        elif self.parameters is None or len(self.parameters) == 0:
            raise ValueError("can't infer jacobians; no parameters specified")
        logging.info("Taking the component jacobians")
        jacobians = gradient.jacobian(self.components, self.parameters)
        jacobian_map = OrderedDict(equizip(self.parameters, jacobians))
        logging.info("The component jacobian computation graph is built")
        return jacobian_map

def print_attributions(algorithm):
    param, hist = zip(*algorithm.attributions.items())
    for pindex in range(0, len(hist)):
        allvals = hist[pindex].get_value()
        pvals = param[pindex].get_value()
        num_classes = allvals.shape[0]
        # if np.prod(allvals.shape[1:]) <= 700:
        #     allvals = np.reshape(allvals, (allvals.shape[0], 1, -1))
        #     pvals = np.reshape(pvals, (1, -1))
        if (hist[pindex].tag.for_parameter.name == 'W' and isinstance(
              hist[pindex].tag.for_parameter.tag.annotations[0], Linear)):
            allvals = np.transpose(allvals, (0, 2, 1))
            pvals = np.transpose(pvals, (1, 0))
        else:
            allvals = np.reshape(allvals, allvals.shape[0:2] + (-1,))
            pvals = np.reshape(pvals, (pvals.shape[0], -1))
        for nindex in range(0, allvals.shape[1]):
            vals = allvals[:,nindex,:]
            name = ('unit %d' % nindex) if allvals.shape[1] > 1 else 'units'
            print('Attribution for parameter %s for layer %s %s' % (
                hist[pindex].tag.for_parameter.name,
                hist[pindex].tag.for_parameter.tag.annotations[0].name,
                name))
            # Individual weight histograms - commented out.
            if False:
                svals = np.sort(vals, axis=0).reshape((vals.shape[0], -1))
                sinds = np.argsort(vals, axis=0).reshape((vals.shape[0], -1))
                for j in range(svals.shape[1]):
                    print('Sorted hist for weight', j, pvals[nindex, j])
                    limit = max(abs(svals[:,j]))
                    for k in range(svals.shape[0]):
                        n = int(np.nan_to_num(32 * svals[k, j] / limit))
                        if n < 0:
                            s = (32 + n) * ' ' + (-n) * '#'
                        else:
                            s = 32 * ' ' + (n + 1) * '#'
                        print(s, svals[k, j], sinds[k, j])

            bounds = sorted(zip(
                vals.argmin(axis=0).flatten(),
                vals.argmax(axis=0).flatten()))
            bc = defaultdict(int)
            for b in bounds:
                if b[0] != b[1]:
                    bc[b] += 1
            for x in range(num_classes):
                printed = False
                for y in range(num_classes):
                    amt = bc[(x, y)]
                    if amt:
                        print('%d -> %d:%s %d' %
                            (y, x, '#' * int(4 * np.log2(amt)), amt))
                        printed = True
                if printed:
                    print()
