import numpy as np
from collections import defaultdict
import torch
from torch.nn.utils import parameters_to_vector
import torch.nn.functional as F
from typing import *
import copy
import math
import pdb

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

class AuxMetricsLogger(BaseCallback):

    # log useful *additional* metrics to the learner object
    # (in addition to the usual performance metrics such as training loss, val loss and val acc)
    # save metrics as a dictionary of nested or unnested lists
    # nested if metric is logged on a per-batch basis; unnested if on a per-epoch basis

    def __init__(self, results_folder=None): 

        self.aux_epoch_metrics = defaultdict(list)
        self.results_folder = results_folder # has trial info

        # self.prev_grad = None
        # self.prev_grads = torch.Tensor().cpu()
        # self.model_params = torch.Tensor().cpu() # is this necessary? maybe for regularization, or edge activation statistics

        # self.prev_cosine_sims = []

    def on_epoch_start(self, learner):  

        self.new_epoch = True

        # self.batch_metrics stores the batch-wise metrics in one epoch
        # each key is a metric name, value is the list of metric values for each batch
        self.batch_metrics = defaultdict(list)

        self.epoch_accum_grad = torch.Tensor().cpu()      
        #TODO: reset self.prev_grad and self.prev_grads and self.prev_cosine_sims here  
        # check correctness
        self.prev_grad = None
        self.prev_grads = torch.Tensor().cpu()
        self.prev_cosine_sims = []
    
    def on_after_backward(self, learner):

        signals = self.batch_gradient_signals(learner)

        # append batch-wise signals to self.batch_metrics        
        for k, v in signals.items():
            self.batch_metrics[k].append(v)
            # if self.new_epoch or self.batch_metrics[k] == []:
            #     self.batch_metrics[k].append([v])
            # else:
            #     self.batch_metrics[k][-1].append(v)

        # log to neptune
        if learner.run is not None: 
            for k, v in signals.items():
                if not np.isnan(v):
                    learner.run['signals/batch/' + k].log(v)

    # return dictionary of additional signals
    def batch_gradient_signals(self, learner):

        # problem: only epoch 0 batch 0 returns nonzero signal

        batch_grad = parameters_to_vector(j.grad for j in learner.model.parameters()).cpu().detach().clone()
        norm_batch_grad = torch.linalg.norm(batch_grad).item()
        # norm_aux_batch_grad = torch.linalg.norm(parameters_to_vector(j.grad for j in learner.aux_model.parameters()).cpu()).item()
        # diff_sq_norm_main_aux_batch_grads = norm_batch_grad**2 - norm_aux_batch_grad**2

        # TODO:
        # reset self.prev_grad and self.prev_grads

        if self.prev_grad is not None:
            norm_change_batch_grad = torch.linalg.norm(batch_grad - self.prev_grad).item()
            denoise_signal_1 = norm_batch_grad**2 - 1/2 * norm_change_batch_grad**2
            #denoise_signal_2 = denoise_signal_1 / norm_batch_grad**2
            cosine_sim_batch_grad = F.cosine_similarity(self.prev_grad, batch_grad, dim=0).item()
            self.prev_cosine_sims.append(cosine_sim_batch_grad)

        else:
            # norm_change_batch_grad, denoise_signal_1, denoise_signal_2 = np.nan, np.nan, np.nan
            norm_change_batch_grad, denoise_signal_1 = np.nan, np.nan
            cosine_sim_batch_grad = np.nan
        
        running_avg_of_cosine_sim = np.mean(self.prev_cosine_sims[-learner.base_config['running_avg_window_size']:])

        self.prev_grad = batch_grad
        if self.epoch_accum_grad.nelement() == 0:
            self.epoch_accum_grad = batch_grad
        else:
            self.epoch_accum_grad.add_(batch_grad)

        self.prev_grads = torch.cat((self.prev_grads, batch_grad.unsqueeze(0))).detach().clone()[-learner.base_config['running_avg_window_size']:, :]
        norm_of_running_avg_of_batch_grad = torch.linalg.norm(torch.mean(self.prev_grads, dim = 0)).item()
        running_avg_of_norm_batch_grad = torch.mean(torch.linalg.norm(self.prev_grads, dim=1)).item()
        running_avg_of_squared_norm_batch_grad = torch.mean(torch.linalg.norm(self.prev_grads, dim=1)**2).item()

        denoise_signal_3 = learner.base_config['running_avg_window_size'] * norm_of_running_avg_of_batch_grad**2 - running_avg_of_squared_norm_batch_grad


        # pdb.set_trace()

        # create return values dictionary
        return_metrics = {'norm_batch_grad': norm_batch_grad,\
                # 'norm_aux_batch_grad': norm_aux_batch_grad,\
                # 'diff_sq_norm_main_aux_batch_grads': diff_sq_norm_main_aux_batch_grads,\
                'norm_change_batch_grad': norm_change_batch_grad,\
                'denoise_signal_1': denoise_signal_1,\
                #'denoise_signal_2': denoise_signal_2,\
                #'cosine_sim_batch_grad': cosine_sim_batch_grad,\
                'running_avg_of_cosine_sim_batch_grad': running_avg_of_cosine_sim,\
                'norm_of_running_avg_of_batch_grad': norm_of_running_avg_of_batch_grad,\
                'running_avg_of_norm_batch_grad': running_avg_of_norm_batch_grad,\
                'running_avg_of_squared_norm_batch_grad': running_avg_of_squared_norm_batch_grad,\
                'denoise_signal_3': denoise_signal_3}

        # throw out the nan values
        return_metrics = {k: return_metrics[k] for k in return_metrics if not math.isnan(return_metrics[k])}
        
        return return_metrics

    # TODO: add a time-series-ish epoch signal that characterizes "d metric / d epoch" ?
    # TODO: compute running average of cosine similarity

    def on_train_end(self, learner):    

        norm_epoch_accum_grad = torch.linalg.norm(self.epoch_accum_grad).item()
        self.aux_epoch_metrics['epoch_accum_grad'].append(norm_epoch_accum_grad)
        if learner.run is not None:
            learner.run['metrics/epoch/accum_grad'].log(norm_epoch_accum_grad)

        # TODO: check correctness
        # after training in the current epoch, log the average of batch metrics in this epoch
        for k in self.batch_metrics.keys():
            # print('length of logged batch-wise {} is {}'.format(k, len(self.batch_metrics[k])))
            # print('at epoch {}, log batch metric {}'.format(learner.epoch, k), self.batch_metrics[k])
            self.aux_epoch_metrics[k].append(np.mean(self.batch_metrics[k]))

    def on_val_end(self, learner):

        #self.epoch_metrics['val_loss'].append(learner.current_val_loss)
        #self.epoch_metrics['val_acc'].append(learner.current_val_acc)

        if learner.run is not None:
            learner.run['metrics/epoch/val_loss'].log(learner.current_val_loss)
            learner.run['metrics/epoch/val_acc'].log(learner.current_val_acc)
            learner.run['metrics/epoch/training_loss'].log(learner.current_training_loss)
            learner.run['metrics/epoch/training_time'].log(learner.current_epoch_training_time)


    def on_epoch_end(self, learner):
        # save model if results_folder is specified
        if self.results_folder is not None:
            model_path = self.results_folder + 'model_epoch_' + str(learner.epoch) + '.pth'
            torch.save(learner.model.state_dict(), model_path)
            np.save(self.results_folder + 'logged_metrics.npy', self.aux_epoch_metrics)

        # log the additional metrics to learner
        for k in self.aux_epoch_metrics.keys():
            learner.epoch_metrics[k].append(self.aux_epoch_metrics[k][-1])

class EarlyStopping(BaseCallback):
    def __init__(self, patience, warmup, metric = 'val_loss', tolerance_thresh = 0, to_minimize = True):
        super().__init__()
        self.metric = metric
        self.patience = patience
        self.warmup = warmup
        self.tolerance_thresh = tolerance_thresh
        self.to_minimize = to_minimize
        #self.best_so_far = np.inf if to_minimize else -np.inf
        self.patience_counter = 0


    def do_early_stopping(self, learner):
        # if no improvement in the last {patience} epochs, stop

        # TODO: add other pruning methods, e.g., median pruning
        
        metric_history = learner.epoch_metrics[self.metric]

        if len(metric_history) < self.warmup:
            return False

        if self.to_minimize:
            best_so_far = min(metric_history)
            if metric_history[-1] > best_so_far + self.tolerance_thresh:
                self.patience_counter += 1
            else:
                self.patience_counter = 0
        else:
            best_so_far = max(metric_history)
            if metric_history[-1] < best_so_far - self.tolerance_thresh:
                self.patience_counter += 1
            else:
                self.patience_counter = 0
        
        if self.patience_counter > self.patience:
            print("No improvement in the last {} epochs, terminating this run at epoch {}.".format(self.patience, len(metric_history)) )
            return True
        else:
            return False

    def on_epoch_end(self, learner):
        # if the monitored metrics got worse set a flag to stop training
        if self.do_early_stopping(learner):
            print('stop training early')
            learner.stop = True

class LRMonitor(BaseCallback):
    pass


# see PyTorch lightning callbacks: 
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#best-practices