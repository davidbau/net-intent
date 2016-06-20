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
from filmstrip import plan_grid
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

class SynpicStatisticsRole(PersistentRole):
    pass

# role for synpic historgram
SYNPIC_STATISTICS = SynpicStatisticsRole()

def _create_synpic_histogram_for(param, pic_size, label_count):
    # The synpic histogram is a 2d-array of pic_size.
    # For a 3d parameter, that ends up being a 5d tensor.
    # For a 1d parameter, that's a 3d tensor.
    shape = param.get_value().shape + (label_count,) + pic_size
    print('param', param, 'gets histogram with shape', shape)
    buf = shared_floatx_zeros(shape)
    buf.tag.for_parameter = param
    add_role(buf, SYNPIC_STATISTICS)
    return buf

def _create_synpic_updates(synpic, jacobian, attributed_pic):
    # Updates synpic which is (units1, units2.., labels, picy, picx)
    # By dotting jacobian: (cases, units1, units2..) and
    # attributed_pic (cases, labels, picy, picx)
    return (synpic, synpic * 0.97 -
            tensor.tensordot(jacobian, attributed_pic, axes=((0,), (0,))))

class SynpicGradientDescent(GradientDescent):
    def __init__(self, synpic_parameters=None,
            case_costs=None, pics=None, case_labels=None,
            batch_size=None, pic_size=None, label_count=None, **kwargs):
        super(SynpicGradientDescent, self).__init__(**kwargs)
        center_val = 0.5
        self.input_pics = pics
        self.case_costs = case_costs
        self.batch_size = batch_size
        self.label_count = label_count
        self.synpic_parameters = synpic_parameters
        self.jacobians = self._compute_jacobians()
        self.synpics = OrderedDict(
          [(param, _create_synpic_histogram_for(param, pic_size, label_count))
                for param in self.synpic_parameters])
        # attributes pics: (cases, picy, picx) to (cases, labels, picy, picx)
        # attributed_pics = tensor.batched_tensordot(
        #     tensor.extra_ops.to_one_hot(case_labels.flatten(), label_count),
        #     pics[:, 0, :, :], axes=0)
        zeroed_pics = pics - 0.5
        focused_pics = zeroed_pics * abs(
                gradient.grad(case_costs.mean(), pics))
        attributed_pics = tensor.batched_tensordot(
            tensor.extra_ops.to_one_hot(
                case_labels.flatten(), label_count),
            focused_pics[:, 0, :, :],
            axes=0)
        self.synpic_updates = OrderedDict(
            [_create_synpic_updates(
                self.synpics[param],
                self.jacobians[param],
                attributed_pics) for param in self.synpic_parameters])
        self.add_updates(self.synpic_updates)

    def _compute_jacobians(self):
        if self.case_costs is None or self.case_costs.ndim == 0:
            raise ValueError("can't infer jacobians; no case_costs specified")
        elif self.synpic_parameters is None or len(self.parameters) == 0:
            raise ValueError("can't infer jacobians; no parameters specified")
        logging.info("Taking the synpic jacobians")
        jacobians = gradient.jacobian(self.case_costs, self.synpic_parameters)
        jacobian_map = OrderedDict(equizip(self.synpic_parameters, jacobians))
        logging.info("The synpic jacobian computation graph is built")
        return jacobian_map

    def save_images(self, pattern=None, aspect_ratio=None):
        if pattern is None:
            pattern = '%s_%s_synpic.jpg'
        for param, synpic in self.synpics.items():
            layername = param.tag.annotations[0].name
            paramname = param.name
            allpics = synpic.get_value()
            if len(allpics.shape) == 7:
                allpics = allpics.reshape(
                                (allpics.shape[0], -1) + allpics.shape[-3:])
            has_subunits = len(allpics.shape) == 5
            units = allpics.shape[0]
            subunits = allpics.shape[1] if has_subunits else 1
            unit_width = subunits * self.label_count + 1
            column_height, column_count = plan_grid(units, aspect_ratio,
                    allpics.shape, (1, unit_width))
            filmstrip = Filmstrip(image_shape=allpics.shape[-2:],
                grid_shape=(column_height, column_count * unit_width))

            for unit in range(units):
                col, row = divmod(unit, column_height)
                filmstrip.set_text((row, col * unit_width), "%d" % unit)
                for subunit in range(subunits):
                    if has_subunits:
                        im = allpics[unit, subunit, :, :, :]
                    else:
                        im = allpics[unit, :, :, :]
                    imin = im.min()
                    imax = im.max()
                    scale = (imax - imin) / 2
                    im = im / (scale + 1e-9) + 0.5
                    for label in range(self.label_count):
                        filmstrip.set_image((row, label + 1 +
                                col * unit_width + subunit * self.label_count),
                                im[label, :, :])
            filmstrip.save(pattern % (layername, paramname))

    def save_composite_image(self,
                title=None, graph=None, graph_len=None,
                filename=None, aspect_ratio=None, unit_order=None):
        if filename is None:
            pattern = 'synpic.jpg'
        unit_count = 0
        layer_count = 0
        if graph is not None:
            unit_count += 4 # TODO: make configurable
        if title is not None:
            unit_count += 1
        for param, synpic in self.synpics.items():
            allpics = synpic.get_value()
            if len(allpics.shape) != 4:
                raise NotImplementedError('%s has %s dimensions' % (
                    param.name, allpics.shape))
            layer_count += 1
            unit_count += allpics.shape[0]
        unit_width = self.label_count + 1
        column_height, column_count = plan_grid(unit_count + layer_count,
                aspect_ratio, allpics.shape, (1, unit_width))
        filmstrip = Filmstrip(image_shape=allpics.shape[-2:],
            grid_shape=(column_height, column_count * unit_width))
        pos = 0
        if graph is not None:
            col, row = divmod(pos, column_height)
            filmstrip.plot_graph((row, col * unit_width + 1),
                (4, unit_width - 1),
                graph, graph_len)
            pos += 4
        if title is not None:
            col, row = divmod(pos, column_height)
            filmstrip.set_text((row, col * unit_width + unit_width // 2),
                    title)
            pos += 1
        for param, synpic in self.synpics.items():
            layername = param.tag.annotations[0].name
            allpics = synpic.get_value()
            units = allpics.shape[0]
            col, row = divmod(pos, column_height)
            filmstrip.set_text((row, col * unit_width + unit_width // 2),
                    layername)
            pos += 1
            if unit_order:
                ordering = unit_order[layername]
            else:
                ordering = range(units)
            for unit in ordering:
                col, row = divmod(pos, column_height)
                filmstrip.set_text((row, col * unit_width), "%d:" % unit)
                im = allpics[unit, :, :, :]
                imin = im.min()
                imax = im.max()
                scale = (imax - imin) / 2
                im = im / (scale + 1e-9) + 0.5
                for label in range(self.label_count):
                    filmstrip.set_image((row, label + 1 +
                        col * unit_width), im[label, :, :])
                pos += 1
        filmstrip.save(filename)
