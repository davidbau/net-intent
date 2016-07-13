from PIL import Image, ImageFont, ImageDraw
import os
import os.path
import numpy
import pkg_resources
import io

# Figuring how to conform to an aspect ratio:
# uw = unit width, uh = unit height
# r = target aspect ratio
# (width * uw) / (height * uh) <= aspect_ratio
# width * height = units
# This implies:
# width <= sqrt(aspect_ratio * units * uh / uw)
def plan_grid(unit_count, aspect_ratio, image_shape=None,
                unit_shape=None, nearest=False, margin=1):
    if aspect_ratio is None:
        return (unit_count, 1)
    if image_shape is None:
        image_shape = (1, 1)
    if unit_shape is None:
        unit_shape = (1, 1)
    uh = (image_shape[-2] + margin) * unit_shape[-2]
    uw = (image_shape[-1] + margin) * unit_shape[-1]
    ideal_column_count = numpy.sqrt(aspect_ratio * unit_count * uh / uw)
    if nearest:
        column_count = int(numpy.round(ideal_column_count))
    else:
        column_count = int(numpy.floor(ideal_column_count))
    column_count = max(1, column_count)
    column_height = int(numpy.ceil(unit_count / column_count))
    column_count = int(numpy.ceil(unit_count / column_height))
    return (column_height, column_count)

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
        self.fontfile = pkg_resources.resource_filename(__name__,
                "font/OpenSans-Regular.ttf")
        self.draw = ImageDraw.Draw(self.im)

    def corner_from_grid_location(self, grid_location):
        return tuple(reversed(tuple((g * (s + self.margin))
                for g, s in zip(grid_location, self.image_shape))))

    def set_image(self, grid_location, image_data, mask_data=None,
                    negative=None, zeromean=False, unit_range=None,
                    overflow_color=1):
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
                image_data[overflow_color,:,:] += low_data[0,:,:]
                image_data[overflow_color,:,:] -= high_data[0,:,:]
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
        self.im.paste(one_image, self.corner_from_grid_location(grid_location))

    def set_text(self, grid_location, text, size=None, fill='black'):
        if size is None:
            size = int(self.image_shape[0] / 2)
        font = ImageFont.truetype(self.fontfile, size)
        lines = [line.strip() for line in text.split('\n')]
        sizes = [self.draw.textsize(line, font=font) for line in lines]
        xc, yc = self.corner_from_grid_location(grid_location)
        y = (self.image_shape[0] - sum(h for w, h in sizes)) / 2
        bw = self.image_shape[1]
        for line, (w, h) in zip(lines, sizes):
            location = (xc + (bw - w) / 2, yc + y)
            self.draw.text(location, line, font=font, fill=fill)
            y += h

    def plot_graph(self, grid_location, grid_extent, data,
            data_len=None, fill='black'):
        xc, yc = self.corner_from_grid_location(grid_location)
        ys, xs = (g * (s + self.margin)
                for g, s in zip(grid_extent, self.image_shape))
        if data_len is not None:
            dlen = float(data_len)
            data = data[-data_len:]
        else:
            dlen = float(len(d))
        for i, d in enumerate(data):
            if d is not None and 0 < d < 1:
                x = i / dlen * xs + xc
                y = (1 - d) * ys + yc
                self.draw.ellipse((x, y, x + 1, y + 1), fill)

    def save(self, filename):
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        #if self.im.size[0] * self.im.size[1] < 640 ** 2:
        opts = { 'subsampling': 0, 'quality': 99 }
        self.im.save(filename, 'JPEG', **opts)

    def save_bytes(self, format='PNG'):
        output = io.BytesIO()
        self.im.save(output, format=format)
        contents = output.getvalue()
        output.close()
        return contents
