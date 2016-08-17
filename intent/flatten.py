from blocks.bricks import application, lazy
from blocks.bricks import Brick

class GlobalAverageFlattener(Brick):
    """Flattens the input by applying spatial averaging for each channel.
    It may be used to pass multidimensional objects like images or feature
    maps of convolutional bricks into bricks which allow only two
    dimensional input (batch, features) like MLP.
    """
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return input_.mean(axis=list(range(2, input_.ndim)))

