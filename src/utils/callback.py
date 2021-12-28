import numpy as np
import torch
from typing import *


class BaseCallback():
    """
    Base class for callbacks
    I'd like this to handle:
    1. logging metrics and time/resource consumption every epoch and/or every batch
    2. early stopping: if new values do not surpass the best so far for a while --> not improving 
    3. learning rate scheduling
    4. gradient clipping
    5. gradient accumulation
    6. saving model checkpoints (each epoch, best so far)

    NOTE: when inheriting, can pass in different arguments for different functions
    """

    def __init__(self): pass
    def on_epoch_start(self): pass
    def on_epoch_end(self): pass
    def on_train_batch_start(self): pass
    def on_train_batch_end(self): pass
    def on_train_end(self): pass
    def on_val_batch_start(self): pass
    def on_val_batch_end(self): pass
    def on_val_end(self): pass
    def on_loss_begin(self): pass
    def on_loss_end(self): pass
    def on_step_begin(self): pass
    def on_step_end(self): pass

class EarlyStopping(BaseCallback):
    def __init__(self, metric, patience, tolerance_thresh, improvement_thresh = 0, to_minimize = True):
        super().__init__()
        self.metric = metric
        self.patience = patience
        self.improvement_thresh = improvement_thresh
        self.tolerance_thresh = tolerance_thresh
        self.to_minimize = to_minimize
        self.best_so_far = np.inf if to_minimize else -np.inf

    def do_early_stopping(self, learner):
        # if no improvement in the last {patience} epochs, stop
        # e.g., if minimizing, then loss(t) - loss(t-1) over last {patience} t values
        # sum to something larger than {tolerance_thresh}

        # TODO: for this, need a log of the history of metric values
        pass


    def on_epoch_end(self, learner,  **kwargs):
        # if the monitored metrics got worse set a flag to stop training
        #if some_fct(learner.last_metrics): return {'stop_training': True}

        if self.do_early_stopping():
            learner.stop = True
        else:
            learner.stop = False

        # TODO: think where to assign the "to_stop" attribute

class SignalLogger(BaseCallback):
    # log signals on batch end & on epoch end
    # may need to combine 
    def on_epoch_begin(self): 
        # initialize empty lists for logging; create directories
        pass

    def on_train_batch_begin(self, batch_idx):
        # save batch checkpoints
        return super().on_batch_begin()

    def on_train_epoch_end(self, batch_idx):
        # log useful metrics
        pass


class LRMonitor(BaseCallback):
    pass


# see PyTorch lightning callbacks: 
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#best-practices
class ModelSaver(BaseCallback):
    def on_epoch_end(self):
        return super().on_epoch_end()

# save model
#model_path = results_folder + 'model_epoch_' + str(epoch) + '.pth'
#torch.save(learner.model.state_dict(), model_path)