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

def _create_gradpic_histogram_for(param, pic_size):
    # The gradpic histogram is a 2d-array of pic_size.
    # For a 3d parameter, that ends up being a 5d tensor.
    # For a 1d parameter, that's a 3d tensor.
    shape = param.get_value().shape + pic_size
    print('param', param, 'gets histogram with shape', shape)
    buf = shared_floatx_zeros(param.get_value().shape + pic_size)
    buf.tag.for_parameter = param
    add_role(buf, GRADPIC_STATISTICS)
    return buf

def _create_gradpic_updates(gradpic, jacobian, pic):
    from theano.printing import Print
    # Gradpic: (units1, units2, picy, picx)
    # pic: (cases, picy, pix)
    # jacobian: (cases, units1, units2)
    print('gradpic', gradpic.ndim, 'pic', pic.ndim, 'jacobian', jacobian.ndim)
    # padded_pic = Print('padded_pic', attrs=['shape'])(tensor.shape_padleft(pic, n_ones=jacobian.ndim))
    # padded_jacobian = Print('padded_jacobian', attrs=['shape'])(tensor.shape_padright(jacobian, n_ones=pic.ndim))
    return (gradpic,
            gradpic + tensor.tensordot(jacobian, pic, axes=((0,), (0,))))

class GradpicGradientDescent(GradientDescent):
    def __init__(self, case_costs=None, pics=None,
                    batch_size=None, pic_size=None, **kwargs):
        super(GradpicGradientDescent, self).__init__(**kwargs)
        center_val = 0.5
        self.input_pics = pics
        self.case_costs = case_costs
        self.batch_size = batch_size
        self.jacobians = self._compute_jacobians()
        self.gradpics = OrderedDict(
            [(param, _create_gradpic_histogram_for(param, pic_size))
                for param in self.parameters])
        self.gradpic_updates = OrderedDict(
            [_create_gradpic_updates(
                self.gradpics[param],
                self.jacobians[param],
                pics[:,0,:,:]) for param in self.parameters])
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
            if len(allpics.shape) == 6:
                allpics = allpics.reshape(
                                (allpics.shape[0], -1) + allpics.shape[-2:])
            has_subunits = len(allpics.shape) == 4
            units = allpics.shape[0]
            subunits = allpics.shape[1] if has_subunits else 1
            filmstrip = Filmstrip(image_shape=allpics.shape[-2:],
                            grid_shape=(units, subunits))
            for unit in range(units):
                for subunit in range(subunits):
                    if has_subunits:
                        im = allpics[unit, subunit, :, :]
                    else:
                        im = allpics[unit, :, :]
                    imin = im.min()
                    imax = im.max()
                    im = (im - imin) / (imax - imin + 1e-9)
                    filmstrip.set_image((subunit, unit), im)
            filmstrip.save('%s_%s_gradpic.jpg' % (layername, paramname))

