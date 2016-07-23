from PIL import Image, ImageFont, ImageDraw
import os
import os.path
import numpy
import pkg_resources
import io

class Scatter:
    def __init__(self, shape=None, unit_shape=None,
            background='white'):
        self.shape = shape or (512, 512)
        self.margin = tuple(d // 2 for d in (unit_shape or (28, 28)))
        self.background = background
        self.im = Image.new('RGB',
                tuple(reversed(self.shape)), self.background)
        self.fontfile = pkg_resources.resource_filename(__name__,
                "font/OpenSans-Regular.ttf")
        self.draw = ImageDraw.Draw(self.im, mode='RGBA')

    def corner_from_location(self, location, image_size=None):
        return tuple(reversed(tuple(int(m + (s - m * 2) * l - i / 2) for s, m, l, i in
            zip(self.shape, self.margin, location, image_size or (0, 0)))))

    def set_image(self, location, image_data, fill=None):
        if fill is None:
            fill = (0, 0, 0)
        if len(image_data.shape) == 3 and image_data.shape[0] == 1:
            image_data = image_data[0, :, :]
        if len(image_data.shape) == 3:
            data = (image_data.transpose((1, 2, 0))
                ).astype(numpy.uint8).tostring()
            one_image = Image.frombytes('RGB', image_data.shape[-2:], data)
            self.im.paste(one_image, self.corner_from_location(location,
                image_size=image_data.shape[-2:]))
        elif len(image_data.shape) == 2:
            data = (image_data[:, :, None] *
                     numpy.asarray([0, 0, 0, 1])[None, None, :] +
                     numpy.asarray(fill + (0,))[None, None, :]
                     ).astype(numpy.uint8).tostring()
            one_image = Image.frombytes('RGBA', image_data.shape[-2:], data)
            corner = self.corner_from_location(
                    location, image_size=image_data.shape[-2:])
            box = corner + tuple(c + s for c, s in zip(corner, image_data.shape[-2:]))
            # self.draw.bitmap([corner], one_image)
            self.im.paste(one_image, corner, mask=one_image)

    def set_text(self, location, text, size=None, fill='black'):
        if size is None:
            size = int(self.image_shape[0] / 2)
        font = ImageFont.truetype(self.fontfile, size)
        lines = [line.strip() for line in text.split('\n')]
        sizes = [self.draw.textsize(line, font=font) for line in lines]
        xc, yc = self.corner_from_location(location)
        y = yc - sum(h for w, h in sizes) / 2
        bw = self.image_shape[1]
        for line, (w, h) in zip(lines, sizes):
            location = (xc - w / 2, yc + y)
            self.draw.text(loc, line, font=font, fill=fill)
            y += h

    def draw_line(self, locations, fill='black', width=2):
        x0, y0 = self.corner_from_location(locations[0])
        for loc in locations[1:]:
            x1, y1 = self.corner_from_location(loc)
            self.draw.line((x0, y0, x1, y1), fill=fill, width=width)
            x0, y0 = x1, y1

    def plot_scatter(self, corpus=None, locations=None):
        offset = locations[: ,:2].min(axis=0)
        scale = locations[:, :2].max(axis=0) - offset
        points = (locations[:, :2] - offset) / scale
        if locations.shape[1] > 2:
            ordering = locations[:, 2].argsort()
            if locations.shape[1] == 4:
                numbuckets = 100
                minbucket = locations[:, 3].min()
                bucketscale = ((locations[:, 3].max() - minbucket) / numbuckets
                        ) * (1 + 1e-9)
            else:
                numbuckets = 1
                minbucket = bucketscale = None
            longest = None
            for x in range(numbuckets):
                if numbuckets is 1:
                    bucket = ordering
                else:
                    bucket = [o for o in ordering
                        if x <= (locations[o, 3] - minbucket) / bucketscale < x + 1]
                if len(bucket):
                    color = (int(255 * x / numbuckets), 0,
                            int(255 * (1 - x / numbuckets)))
                    self.draw_line([points[o] for o in bucket], fill=color + (20,))
                    for o in bucket:
                        self.set_image(points[o], corpus.get_data(request=o)[0][0],
                                fill=color)
                    if not longest or len(longest) < len(bucket):
                        longest = bucket
            if longest:
                self.draw_line([points[o] for o in longest],
                    fill=(255, 255, 0, 128))
        else:
            for i in range(len(points)):
                self.set_image(points[i], corpus.get_data(request=i)[0][0])

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
