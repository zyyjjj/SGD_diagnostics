import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
import torch.nn.functional as F
from typing import *
import copy
from .get_memory_info import get_memory_info 

class BaseCallback():
    """
    Base class for callbacks
    I'd like this to handle:
    1. [done] logging metrics and time/resource consumption every epoch and/or every batch
    2. [done] early stopping: if new values do not surpass the best so far for a while --> not improving --Are there other ways of pruning, e.g., median pruning?
    3. learning rate scheduling
    4. gradient clipping
    5. gradient accumulation
    6. [done] saving model checkpoints (each epoch, best so far)

    NOTE: when inheriting, can pass in different arguments for different functions
    """

    def __init__(self): pass
    def on_epoch_start(self, learner): pass
    def on_epoch_end(self, learner): pass
    def on_train_batch_start(self, learner): pass
    def on_after_backward(self, learner): pass
    def on_train_batch_end(self, learner): pass
    def on_train_end(self, learner): pass
    def on_val_batch_start(self, learner): pass
    def on_val_batch_end(self, learner): pass
    def on_val_end(self, learner): pass

class MetricsLogger(BaseCallback):

    # log useful metrics
    # save metrics as a dictionary of nested or unnested lists
    # nested if metric is logged on a per-batch basis; unnested if on a per-epoch basis

    def __init__(self, results_folder): 

        self.logged_metrics = {}
        self.results_folder = results_folder # has trial info

        self.prev_grad = None
        self.prev_grads = torch.Tensor().cpu()
        # self.model_params = torch.Tensor().cpu() # is this necessary? maybe for regularization, or edge activation statistics

    def on_epoch_start(self, learner):  

        self.new_epoch = True
        self.epoch_accum_grad = torch.Tensor().cpu()        
    
    def on_after_backward(self, learner):

        signals = self.batch_gradient_signals(learner)

        # append batch-wise signals to self.logged_metrics        
        for k, v in signals.items():
            if k in self.logged_metrics.keys():
                #print(k, self.logged_metrics[k])
                if self.new_epoch:
                    self.logged_metrics[k].append([v])
                else:
                    self.logged_metrics[k][-1].append(v)
            else:
                self.logged_metrics[k] = [[v]]
        self.new_epoch = False

        # log to neptune
        for k, v in signals.items():
            if not np.isnan(v):
                learner.run['signals/batch/' + k].log(v)

    # return dictionary of additional signals
    def batch_gradient_signals(self, learner):

        batch_grad = parameters_to_vector(j.grad for j in learner.model.parameters()).cpu().detach().clone()
        norm_batch_grad = torch.linalg.norm(batch_grad).item()
        norm_aux_batch_grad = torch.linalg.norm(parameters_to_vector(j.grad for j in learner.aux_model.parameters()).cpu()).item()
        diff_sq_norm_main_aux_batch_grads = norm_batch_grad**2 - norm_aux_batch_grad**2

        if self.prev_grad is not None:
            norm_change_batch_grad = torch.linalg.norm(batch_grad - self.prev_grad).item()
            denoise_signal_1 = norm_batch_grad**2 - 1/2 * norm_change_batch_grad**2
            #denoise_signal_2 = denoise_signal_1 / norm_batch_grad**2
            cosine_sim_batch_grad = F.cosine_similarity(self.prev_grad, batch_grad, dim=0).item()
        else:
            # norm_change_batch_grad, denoise_signal_1, denoise_signal_2 = np.nan, np.nan, np.nan
            norm_change_batch_grad, denoise_signal_1 = np.nan, np.nan
            cosine_sim_batch_grad = np.nan

        self.prev_grad = batch_grad
        if self.epoch_accum_grad.nelement() == 0:
            self.epoch_accum_grad = batch_grad
        else:
            self.epoch_accum_grad.add_(batch_grad)

        # to handle stuff with prev_gradients:
        #prev_grads = copy.deepcopy(torch.cat((self.prev_grads, batch_grad.unsqueeze(0))))[-learner.base_config['running_avg_window_size']:, :]
        self.prev_grads = torch.cat((self.prev_grads, batch_grad.unsqueeze(0))).detach().clone()[-learner.base_config['running_avg_window_size']:, :]
        norm_of_running_avg_of_batch_grad = torch.linalg.norm(torch.mean(self.prev_grads, dim = 0)).item()
        running_avg_of_norm_batch_grad = torch.mean(torch.linalg.norm(self.prev_grads, dim=1)).item()
        running_avg_of_squared_norm_batch_grad = torch.mean(torch.linalg.norm(self.prev_grads, dim=1)**2).item()

        denoise_signal_3 = learner.base_config['running_avg_window_size'] * norm_of_running_avg_of_batch_grad**2 - running_avg_of_squared_norm_batch_grad

        return {'norm_batch_grad': norm_batch_grad,\
                'norm_aux_batch_grad': norm_aux_batch_grad,\
                'diff_sq_norm_main_aux_batch_grads': diff_sq_norm_main_aux_batch_grads,\
                'norm_change_batch_grad': norm_change_batch_grad,\
                'denoise_signal_1': denoise_signal_1,\
                #'denoise_signal_2': denoise_signal_2,\
                'cosine_sim_batch_grad': cosine_sim_batch_grad,\
                'norm_of_running_avg_of_batch_grad': norm_of_running_avg_of_batch_grad,\
                'running_avg_of_norm_batch_grad': running_avg_of_norm_batch_grad,\
                'running_avg_of_squared_norm_batch_grad': running_avg_of_squared_norm_batch_grad,\
                'denoise_signal_3': denoise_signal_3}

    def on_train_end(self, learner):
        
        for k in ['training_loss', 'training_time', 'epoch_accum_grad']:
            if k not in self.logged_metrics.keys():
                self.logged_metrics[k] = []       

        self.logged_metrics['training_loss'].append(learner.current_training_loss)
        self.logged_metrics['training_time'].append(learner.current_epoch_training_time)
        norm_epoch_accum_grad = torch.linalg.norm(self.epoch_accum_grad)
        self.logged_metrics['epoch_accum_grad'].append(norm_epoch_accum_grad)

        learner.run['metrics/epoch/training_loss'].log(learner.current_training_loss)
        learner.run['metrics/epoch/training_time'].log(learner.current_epoch_training_time)
        learner.run['metrics/epoch/accum_grad'].log(norm_epoch_accum_grad)

    def on_val_end(self, learner):

        for k in ['val_loss', 'val_acc']:
            if k not in self.logged_metrics.keys():
                self.logged_metrics[k] = []

        self.logged_metrics['val_loss'].append(learner.current_val_loss)
        self.logged_metrics['val_acc'].append(learner.current_val_acc)

        learner.run['metrics/epoch/val_loss'].log(learner.current_val_loss)
        learner.run['metrics/epoch/val_acc'].log(learner.current_val_acc)


    def on_epoch_end(self, learner):
        # save model
        model_path = self.results_folder + 'model_epoch_' + str(learner.epoch) + '.pth'
        torch.save(learner.model.state_dict(), model_path)


        # BAD PRACTICE to avoid in the future: if learner.logged_perf_metrics is only updated here,
        # then this creates dependency between the MetricsLogger class and the EarlyStopping class
        # for k in learner.logged_performance_metrics.keys():
        #     learner.logged_performance_metrics[k].append(self.logged_metrics[k][-1])
        
        np.save(self.results_folder + 'logged_metrics.npy', self.logged_metrics)
        # np.save(self.results_folder + 'logged_performance_metrics.npy', learner.logged_performance_metrics)

    

class EarlyStopping(BaseCallback):
    def __init__(self, patience, warmup, metric = 'val_loss', tolerance_thresh = 0, to_minimize = True):
        super().__init__()
        self.metric = metric
        self.patience = patience
        self.warmup = warmup
        self.tolerance_thresh = tolerance_thresh
        self.to_minimize = to_minimize
        self.best_so_far = np.inf if to_minimize else -np.inf


    def do_early_stopping(self, learner):
        # if no improvement in the last {patience} epochs, stop
        # e.g., if minimizing, then loss(t) - loss(t-1) over last {patience} t values
        # sum to something larger than {tolerance_thresh}

        # TODO: add other pruning methods, e.g., median pruning

        metric_history = learner.logged_performance_metrics[self.metric]

        if len(metric_history) < self.warmup:
            return False
        
        if self.to_minimize:
            if metric_history[-1] - metric_history[-(self.patience+1)] > self.tolerance_thresh:
                print("No improvement in the last {} epochs, terminating this run.".format(self.patience) )
                return True
        else:
            if metric_history[-1] - metric_history[-(self.patience+1)] < -self.tolerance_thresh:
                print("No improvement in the last {} epochs, terminating this run.".format(self.patience) )
                return True
        
        return False

    def on_epoch_end(self, learner):
        # if the monitored metrics got worse set a flag to stop training

        if self.do_early_stopping(learner):
            learner.stop = True
class LRMonitor(BaseCallback):
    pass


# see PyTorch lightning callbacks: 
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#best-practices