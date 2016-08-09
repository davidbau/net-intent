from blocks.extensions import SimpleExtension
import numpy

class EpochSchedule(SimpleExtension):
    def __init__(self, parameter, schedule, **kwargs):
        kwargs.setdefault("before_epoch", True)
        super(EpochSchedule, self).__init__(**kwargs)
        self.parameter = parameter
        self.schedule = schedule

    def do(self, which_callback, *args):
        epochs_done = self.main_loop.log.status['epochs_done']
        for begin, value in reversed(self.schedule):
            if epochs_done >= begin:
                self.parameter.set_value(value)
                return

class EpochExponentiation(SimpleExtension):
    def __init__(self, parameter, exponent, **kwargs):
        kwargs.setdefault("before_epoch", True)
        super(EpochExponentiation, self).__init__(**kwargs)
        self.parameter = parameter
        self.exponent = exponent

    def do(self, which_callback, *args):
        value = self.parameter.get_value()
        self.parameter.set_value(numpy.asarray(
                value ** self.exponent, dtype=self.parameter.dtype))
