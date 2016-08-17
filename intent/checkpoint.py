from blocks.extensions.saveload import Checkpoint

class EpochCheckpoint(Checkpoint):
    def do(self, callback_name, *args):
        if '%d' in self.path:
            epoch_path = self.path % self.main_loop.status['epochs_done']
        else:
            epoch_path = self.path
        super().do(callback_name, epoch_path)

