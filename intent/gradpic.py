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
from filmstrip import Filmstrip
from picklable_itertools.extras import equizip
import numpy

class CasewiseCrossEntropy(Brick):
    """By-case cross entropy.

    Outputs a vector with cross entropy separated out to cases.
    """
    @application(outputs=["case_costs"])
    def apply(self, y, y_hat):
        # outputs a vector with cross entropy for each row
        return tensor.nnet.crossentropy_categorical_1hot(y_hat, y)

class GradpicStatisticsRole(PersistentRole):
    pass

# role for gradpic historgram
GRADPIC_STATISTICS = GradpicStatisticsRole()

def _create_gradpic_histogram_for(param, pic_size, label_count):
    # The gradpic histogram is a 2d-array of pic_size.
    # For a 3d parameter, that ends up being a 5d tensor.
    # For a 1d parameter, that's a 3d tensor.
    shape = param.get_value().shape + (label_count,) + pic_size
    print('param', param, 'gets histogram with shape', shape)
    buf = shared_floatx_zeros(shape)
    buf.tag.for_parameter = param
    add_role(buf, GRADPIC_STATISTICS)
    return buf

def _create_gradpic_updates(gradpic, jacobian, attributed_pic):
    # Updates gradpic which is (units1, units2.., labels, picy, picx)
    # By dotting jacobian: (cases, units1, units2..) and
    # attributed_pic (cases, labels, picy, picx)
    return (gradpic, gradpic -
            tensor.tensordot(jacobian, attributed_pic, axes=((0,), (0,))))

class GradpicGradientDescent(GradientDescent):
    def __init__(self, case_costs=None, pics=None, case_labels=None,
                    batch_size=None, pic_size=None, label_count=None, **kwargs):
        super(GradpicGradientDescent, self).__init__(**kwargs)
        center_val = 0.5
        self.input_pics = pics
        self.case_costs = case_costs
        self.batch_size = batch_size
        self.label_count = label_count
        self.jacobians = self._compute_jacobians()
        self.gradpics = OrderedDict(
          [(param, _create_gradpic_histogram_for(param, pic_size, label_count))
                for param in self.parameters])
        # attributes pics: (cases, picy, picx) to (cases, labels, picy, picx)
        attributed_pics = tensor.batched_tensordot(
            tensor.extra_ops.to_one_hot(case_labels.flatten(), label_count),
            pics[:, 0, :, :], axes=0)
        attributed_pics = attributed_pics
        self.gradpic_updates = OrderedDict(
            [_create_gradpic_updates(
                self.gradpics[param],
                self.jacobians[param],
                attributed_pics) for param in self.parameters])
        self.add_updates(self.gradpic_updates)

    def _compute_jacobians(self):
        if self.case_costs is None or self.case_costs.ndim == 0:
            raise ValueError("can't infer jacobians; no case_costs specified")
        elif self.parameters is None or len(self.parameters) == 0:
            raise ValueError("can't infer jacobians; no parameters specified")
        logging.info("Taking the gradpic jacobians")
        jacobians = gradient.jacobian(self.case_costs, self.parameters)
        jacobian_map = OrderedDict(equizip(self.parameters, jacobians))
        logging.info("The gradpic jacobian computation graph is built")
        return jacobian_map

    def save_images(self):
        for param, gradpic in self.gradpics.items():
            layername = param.tag.annotations[0].name
            paramname = param.name
            allpics = gradpic.get_value()
            if len(allpics.shape) == 7:
                allpics = allpics.reshape(
                                (allpics.shape[0], -1) + allpics.shape[-3:])
            has_subunits = len(allpics.shape) == 5
            units = allpics.shape[0]
            subunits = allpics.shape[1] if has_subunits else 1
            filmstrip = Filmstrip(image_shape=allpics.shape[-2:],
                            grid_shape=(units, subunits * self.label_count))
            for unit in range(units):
                for subunit in range(subunits):
                    if has_subunits:
                        im = allpics[unit, subunit, :, :, :]
                    else:
                        im = allpics[unit, :, :, :]
                    imin = im.min()
                    imax = im.max()
                    im = (im - imin) / (imax - imin + 1e-9)
                    for label in range(self.label_count):
                        filmstrip.set_image(
                                (subunit * self.label_count + label, unit),
                                im[label, :, :])
            filmstrip.save('%s_%s_gradpic.jpg' % (layername, paramname))

