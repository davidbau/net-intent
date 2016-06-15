from PIL import Image
import os
import os.path
import numpy

class Filmstrip:
    def __init__(self, image_shape=None, grid_shape=None,
                    margin=1, background='white'):
        self.image_shape = image_shape
        self.grid_shape = grid_shape
        self.margin = margin
        self.background = background
        self.im = Image.new('RGB',
            ((self.image_shape[1] + margin) * self.grid_shape[1] - margin,
             (self.image_shape[0] + margin) * self.grid_shape[0] - margin),
            self.background)

    def set_image(self, grid_location, image_data, mask_data=None,
                    negative=None, zeromean=False, unit_range=None):
        if unit_range is None and not numpy.issubdtype(
                        image_data.dtype, numpy.integer):
            unit_range = True
        if len(image_data.shape) == 2:
            image_data = image_data[numpy.newaxis, :, :]
        if negative is None:
            negative = image_data.shape[0] == 1
        if image_data.shape[0] == 1:
            if unit_range:
                high_data = (image_data - 1).clip(0, 1)
                low_data = (- image_data).clip(0, 1)
                main_data = image_data.clip(0, 1)
                image_data = numpy.tile(main_data, (3, 1, 1))
                image_data[1,:,:] += low_data[0,:,:]
                image_data[1,:,:] -= high_data[0,:,:]
                # image_data[1,:,:] -= high_data[0,:,:]
            else:
                image_data = numpy.tile(image_data, (3, 1, 1))
        if unit_range:
            image_data = numpy.clip(image_data * 255, 0, 255)
        if zeromean:
            image_data = image_data + 128
        if negative:
            image_data = 255 - image_data
        if mask_data is not None:
            image_data = ((image_data.astype(numpy.float) - 128)
                            * mask_data + 128)
        data = (image_data.transpose((1, 2, 0))
            ).astype(numpy.uint8).tostring()
        one_image = Image.frombytes('RGB', self.image_shape, data)
        self.im.paste(one_image, tuple((g * (s + self.margin))
                for g, s in zip(grid_location, self.image_shape)))

    def save(self, filename):
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        #if self.im.size[0] * self.im.size[1] < 640 ** 2:
        opts = { 'subsampling': 0, 'quality': 99 }
        self.im.save(filename, 'JPEG', **opts)
