import numpy as np
import torch


class Callback():

    """
    Base class for callbacks
    I'd like this to handle:
    1. logging metrics and time/resource consumption every epoch and/or every batch
    2. early stopping: if new values to not surpass the best so far for a while --> not improving 
    3. learning rate scheduling
    4. gradient clipping
    5. gradient accumulation
    6. saving model checkpoints (each epoch, best so far)
    """

    def __init__(self, save_best = True): 
        self.save_best = save_best
    def on_train_begin(self): pass
    def on_train_end(self): pass
    def on_epoch_begin(self): pass
    def on_epoch_end(self): 

        # torch.save(modelA.state_dict(), PATH)
        



        pass

    def on_batch_begin(self): pass
    def on_batch_end(self): pass
    def on_loss_begin(self): pass
    def on_loss_end(self): pass
    def on_step_begin(self): pass
    def on_step_end(self): pass



class EarlyStopping(Callback):
    def __init__(self, metric, patience, tolerance_thresh, improvement_thresh = 0, to_minimize = True):
        super().__init__()
        self.metric = metric
        self.patience = patience
        self.improvement_thresh = improvement_thresh
        self.tolerance_thresh = tolerance_thresh
        self.to_minimize = to_minimize
        self.best_so_far = np.inf if to_minimize else -np.inf

    def check_early_stopping(self):
        # if no improvement in the last {patience} epochs, stop
        # e.g., if minimizing, then loss(t) - loss(t-1) over last {patience} t values
        # sum to something larger than {tolerance_thresh}

        # for this, need a log of the history of metric values


    def on_epoch_end(self, last_metrics, **kwargs):
        # if the monitored metrics got worst set a flag to stop training
        if some_fct(last_metrics): return {'stop_training': True}

class SignalLogger(Callback):
    # log signals on batch end & on epoch end
    # may need to combine 
    def on_epoch_begin(self): 
        # initialize empty lists for logging; create directories
        pass