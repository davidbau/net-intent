import logging
import numpy as np
from collections import OrderedDict, defaultdict

from theano import tensor
from theano import gradient

from blocks.bricks import application
from blocks.bricks import Linear
from blocks.bricks.base import Brick
from blocks.extensions import SimpleExtension
from blocks.monitoring.evaluators import DatasetEvaluator
from blocks.roles import PersistentRole
from blocks.roles import add_role
from blocks.utils import shared_floatx_zeros
from filmstrip import Filmstrip
from filmstrip import plan_grid
from picklable_itertools.extras import equizip
import numpy

class ActpicStatisticsRole(PersistentRole):
    pass

# role for actpic historgram
ACTPIC_STATISTICS = ActpicStatisticsRole()

class ActpicExtension(SimpleExtension):
    def __init__(self, actpic_variables=None, pics=None, case_labels=None,
            label_count=None, data_stream=None, rectify=False, **kwargs):
        center_val = 0.5
        self.input_pics = pics
        # self.batch_size = batch_size
        # self.label_count = label_count
        self.actpic_variables = actpic_variables
        # attributes pics: (cases, picy, picx) to (cases, labels, picy, picx)
        # attributed_pics = tensor.batched_tensordot(
        #     tensor.extra_ops.to_one_hot(case_labels.flatten(), label_count),
        #     pics[:, 0, :, :], axes=0)
        zeroed_pics = pics - 0.5
        attributed_pics = tensor.batched_tensordot(
            tensor.extra_ops.to_one_hot(
                case_labels.flatten(), label_count),
            zeroed_pics[:, 0, :, :],
            axes=0)
        self.actpics = [self._create_actpic_image_for(
              name + '_actpic', var, attributed_pics, rectify)
                for name, var in self.actpic_variables.items()]
        self.evaluator = DatasetEvaluator(self.actpics)
        self.data_stream = data_stream
        self.results = None
        super(ActpicExtension, self).__init__(**kwargs)

    def do(self, callback_name, *args):
        self.parse_args(callback_name, args)
        self.results = self.evaluator.evaluate(self.data_stream)

    def _create_actpic_image_for(self, name, var, pics, rectify):
        # var is (cases, unit1, unit2, ...)
        # pics is (cases, labels, picy, picx)
        # output is (unit1, unit2... labels, picy, picx)
        # TODO: for convolutions, use gaussian to take contribution
        # just around the receptive field of the neuron.  For now we
        # just take the whole image as contribution as follows.
        while var.ndim > 2:
            var = var.sum(axis=var.ndim - 1)
        if rectify == -1:
            var = -tensor.nnet.relu(-var)
        elif rectify:
            var = tensor.nnet.relu(var)
        return tensor.tensordot(var, pics, axes=[[0],[0]]).copy(name=name)

    def get_picdata(self):
        result = OrderedDict()
        for name, actpic in self.results.items():
            layername = name[:-7] # trim '_actpic' from the end
            result[layername] = actpic
        return result
