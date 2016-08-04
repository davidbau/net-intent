from blocks.extensions import SimpleExtension

class EpochSchedule(SimpleExtension):
    def __init__(self, parameter, schedule, **kwargs):
        kwargs.setdefault("before_epoch", True)
        super(EpochSchedule, self).__init__(**kwargs)
        self.parameter = parameter
        self.schedule = schedule

    def do(self, which_callback, *args):
        epochs_done = self.main_loop.log.status['epochs_done']
        for begin, value in self.schedule:
            if epochs_done >= begin:
                self.parameter.set_value(value)
                return