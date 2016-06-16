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

class IntpicStatisticsRole(PersistentRole):
    pass

# role for intpic historgram
INTPIC_STATISTICS = IntpicStatisticsRole()

def _create_intpic_histogram_for(param, pic_size, label_count):
    # The pic histogram is a 2d-array of pic_size.
    # For a 3d parameter, that ends up being a 5d tensor.
    # For a 1d parameter, that's a 3d tensor.
    shape = param.get_value().shape + (label_count,) + pic_size
    buf = shared_floatx_zeros(shape)
    buf.tag.for_parameter = param
    add_role(buf, INTPIC_STATISTICS)
    return buf

def _create_gradpic_updates(gradpic, jacobian, attributed_pic):
    # Updates intpic which is (units1, units2.., labels, picy, picx)
    # By dotting jacobian: (cases, units1, units2..) and
    # attributed_pic (cases, labels, picy, picx)
    return (gradpic, gradpic -
            tensor.tensordot(jacobian, attributed_pic, axes=((0,), (0,))))

def _create_intensity_updates(intpic, jacobian, attributed_pic):
    # Updates intpic which is (units1, units2.., labels, picy, picx)
    # By dotting jacobian: (cases, units1, units2..) and
    # attributed_pic (cases, labels, picy, picx)
    return (intpic, intpic + tensor.clip(
                tensor.tensordot(jacobian, attributed_pic, axes=((0,), (0,))),
                0, numpy.inf))

class IntpicGradientDescent(GradientDescent):
    def __init__(self, intpic_parameters=None,
            case_costs=None, pics=None, case_labels=None,
            batch_size=None, pic_size=None, label_count=None, **kwargs):
        super(IntpicGradientDescent, self).__init__(**kwargs)
        center_val = 0.5
        self.input_pics = pics
        self.case_costs = case_costs
        self.batch_size = batch_size
        self.label_count = label_count
        self.intpic_parameters = intpic_parameters
        self.jacobians = self._compute_jacobians()
        self.gradpics = OrderedDict(
          [(param, _create_intpic_histogram_for(param, pic_size, label_count))
                for param in self.intpic_parameters])
        self.intpics = OrderedDict(
          [(param, _create_intpic_histogram_for(param, pic_size, label_count))
                for param in self.intpic_parameters])
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
        self.gradpic_updates = OrderedDict(
            [_create_gradpic_updates(
                self.gradpics[param],
                self.jacobians[param],
                attributed_pics) for param in self.intpic_parameters])
        self.add_updates(self.gradpic_updates)

        intensity_pics = (zeroed_pics *
                gradient.grad(case_costs.mean(), pics))
        attributed_i_pics = tensor.batched_tensordot(
            tensor.extra_ops.to_one_hot(
                case_labels.flatten(), label_count),
            intensity_pics[:, 0, :, :],
            axes=0)

        self.intpic_updates = OrderedDict(
            [_create_intensity_updates(
                self.intpics[param],
                self.jacobians[param],
                attributed_i_pics) for param in self.intpic_parameters])
        self.add_updates(self.intpic_updates)

    def _compute_jacobians(self):
        if self.case_costs is None or self.case_costs.ndim == 0:
            raise ValueError("can't infer jacobians; no case_costs specified")
        elif self.intpic_parameters is None or len(self.parameters) == 0:
            raise ValueError("can't infer jacobians; no parameters specified")
        logging.info("Taking the intpic jacobians")
        jacobians = gradient.jacobian(self.case_costs, self.intpic_parameters)
        jacobian_map = OrderedDict(equizip(self.intpic_parameters, jacobians))
        logging.info("The intpic jacobian computation graph is built")
        return jacobian_map

    def save_images(self, pattern=None, aspect_ratio=None):
        if pattern is None:
            pattern = '%s_%s_intpic.jpg'
        for param, gradpic in self.gradpics.items():
            layername = param.tag.annotations[0].name
            paramname = param.name
            basepics = gradpic.get_value()
            intpics = self.intpics[param].get_value()
            if len(basepics.shape) == 7:
                basepics = basepics.reshape(
                                (basepics.shape[0], -1) + basepics.shape[-3:])
                intpics = intpics.reshape(
                                (intpics.shape[0], -1) + intpics.shape[-3:])
            has_subunits = len(basepics.shape) == 5
            units = basepics.shape[0]
            subunits = basepics.shape[1] if has_subunits else 1
            unit_width = subunits * self.label_count
            column_count = 1
            column_height = units
            if aspect_ratio is not None:
                uh = basepics.shape[-2] + 1
                uw = (basepics.shape[-1] + 1) * unit_width
                column_count = int(numpy.floor(numpy.sqrt(
                    aspect_ratio * units * uh / uw)))
                column_height = units // column_count
            filmstrip = Filmstrip(image_shape=basepics.shape[-2:],
                            grid_shape=(column_height, column_count * unit_width))

            for unit in range(units):
                col, row = divmod(unit, column_height)
                for subunit in range(subunits):
                    if has_subunits:
                        im = basepics[unit, subunit, :, :, :]
                        intensity = intpics[unit, subunit, :, :, :]
                    else:
                        im = basepics[unit, :, :, :]
                        intensity = intpics[unit, :, :, :]
                    imin = im.min()
                    imax = im.max()
                    intscale = intensity.mean() * 8 # TODO: consider using mean
                    scale = max(imax, -imin)
                    im = im / (scale + 1e-15) * 0.2
                    im = im * (1 + intensity / (intscale + 1e-15)) + 0.5
                    for label in range(self.label_count):
                        filmstrip.set_image((row, label +
                                col * unit_width + subunit * self.label_count),
                                im[label, :, :],
                                overflow_color=2)
            filmstrip.save(pattern % (layername, paramname))

