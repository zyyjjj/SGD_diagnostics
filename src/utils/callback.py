import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
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
    def on_after_backward(self): pass
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
        val_loss_history = learner.logged_metrics['val_loss']
        pass


    def on_epoch_end(self, learner,  **kwargs):
        # if the monitored metrics got worse set a flag to stop training
        #if some_fct(learner.last_metrics): return {'stop_training': True}

        if self.do_early_stopping():
            learner.stop = True
        else:
            learner.stop = False

        # TODO: think where to assign the "to_stop" attribute

class MetricsLogger(BaseCallback):

    def __init__(self, results_folder): 
        self.logged_metrics = {}
        self.results_folder = results_folder

    def on_epoch_begin(self): 
        # initialize empty lists for logging; create directories
        
        self.new_epoch = True

        list_running_avg_norm_of_batch_grad = []
        list_norm_of_batch_grad_running_avg = []
        list_training_time = []
        tensor_accum_grad = torch.Tensor().cpu() # TODO: shouldn't save this on the GPU
        tensor_model_params = torch.Tensor().cpu()
        prev_gradient = None
        prev_gradients = torch.Tensor().cpu()  



    def on_train_batch_begin(self, batch_idx):
        # save batch checkpoints
        return super().on_batch_begin()
    
    def on_after_backward(self, learner, run):

        #batch_gradient = parameters_to_vector(j.grad for j in learner.model.parameters()).cpu()
        #aux_batch_gradient = parameters_to_vector(j.grad for j in learner.aux_model.parameters()).cpu()
        #norm_batch_gradient = torch.linalg.norm(batch_gradient).item()
        #norm_aux_batch_gradient = torch.linalg.norm(aux_batch_gradient).item()

        #run['signals/batch/norm_batch_gradient'].log(norm_batch_gradient)
        #run['signals/batch/norm_aux_batch_gradient'].log(norm_aux_batch_gradient)
        #run['signals/batch/squared_diff_batch_gradient'].log(norm_batch_gradient**2 - norm_aux_batch_gradient**2)

        signals = self.batch_gradient_signals(learner)

        # append batch-wise signals to self.logged_metrics        
        for k, v in signals.items():
            if k in self.logged_metrics.keys():
                if self.new_epoch:
                    self.logged_metrics[k].append([v])
                else:
                    self.logged_metrics[k][-1].append(v)
            else:
                self.logged_metrics[k] = [v]
        self.new_epoch = False

        # log to neptune
        for k, v in signals.items():
            run['signals/batch/' + k].log(v)

    def on_train_end(self, training_loss, training_time):
        # log useful metrics
        # save metrics as a dictionary of nested or unnested lists
        # nested if metric is logged on a per-batch basis; unnested if on a per-epoch basis
        self.logged_metrics['training_loss'].append(training_loss)
        self.logged_metrics['training_time'].append(training_time)

    # return dictionary of additional signals
    def batch_gradient_signals(self, learner):
        batch_gradient = parameters_to_vector(j.grad for j in learner.model.parameters()).cpu()
        norm_batch_gradient = torch.linalg.norm(batch_gradient).item()
        norm_aux_batch_gradient = torch.linalg.norm(parameters_to_vector(j.grad for j in learner.aux_model.parameters()).cpu()).item()
        diff_sq_norm_main_aux_batch_gradients = norm_batch_gradient**2 - norm_aux_batch_gradient**2

        # to handle stuff with prev_gradients:
        # first, need to log the gradients (in vector form)
        # also, use self.logged_metrics to query previous gradients to compute running averages


        # sth like self.logged_metrics['gradient_history']
        # a list of tensors, or a list of flattened tensors? turned into np array?
        batch_gradient.numpy()



        return {'norm_batch_gradient': norm_batch_gradient,\
                'norm_aux_batch_gradient': norm_aux_batch_gradient,\
                'diff_sq_norm_main_aux_batch_gradients': diff_sq_norm_main_aux_batch_gradients}

    # TODO: how to pass in kwargs?
    def on_val_end(self, val_loss, val_acc):
        self.logged_metrics['val_loss'].append(val_loss)
        self.logged_metrics['val_acc'].append(val_acc)

    def on_epoch_end(self, learner, epoch):
        # save model
        model_path = self.results_folder + 'model_epoch_' + str(epoch) + '.pth'
        torch.save(learner.model.state_dict(), model_path)

        # merge self.logged_metrics to learner.logged_metrics

    

class LRMonitor(BaseCallback):
    pass


# see PyTorch lightning callbacks: 
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#best-practices