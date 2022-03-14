import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import random, time, copy, os
from collections import defaultdict

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)

    return [o]

def accuracy(outputs, labels):

    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class Learner():
    def __init__(self, hp_config, model, train_ds, val_ds, opt, loss_fn, callbacks, run=None):

        self.model, self.train_ds, self.val_ds = model, train_ds, val_ds
        self.callbacks = listify(callbacks)
        self.base_config, self.opt_kwargs, self.loss_fn_kwargs = hp_config
        self.train_loader = DataLoader(train_ds, int(self.base_config['batch_size']), shuffle=True)
        self.val_loader = DataLoader(val_ds, int(self.base_config['batch_size']))
        self.optimizer = opt(self.model.parameters(), self.base_config['lr'], **self.opt_kwargs)
        self.loss_fn = loss_fn
        self.run = run

        self.stop = False
        self.epoch_metrics = defaultdict(list)
        self.epoch = 0
        self.current_training_loss, self.current_val_loss, self.current_val_acc = 0.0, 0.0, 0.0
        self.current_epoch_training_time = 0

        print('initiated learner with hp config ', hp_config)
        print('training data size: {}, validation data size: {}'.format(len(train_ds), len(val_ds)))

    def fit(self, num_epochs, device):

        for epoch in range(num_epochs):

            self.epoch = epoch

            self.model.train()
            self._evoke_callback('on_epoch_start')

            tic_epoch_start = time.time()

            aux_indices = random.sample(
                range(len(self.train_ds)), 
                min(int(self.base_config['aux_batch_size']), len(self.train_ds))
            )
            auxiliary_ds = Subset(self.train_ds, aux_indices) 
            aux_loader = DataLoader(auxiliary_ds, len(auxiliary_ds))
            for aux_data in aux_loader:
                aux_input, aux_output = aux_data
            aux_input = aux_input.to(device)
            aux_output = aux_output.to(device)
        
            self.current_training_loss = 0.0

            for train_input, train_output in self.train_loader:
                self._evoke_callback('on_train_batch_start')

                # train and log
                train_input, train_output = train_input.to(device), train_output.to(device)
                loss = self.loss_fn(self.model(train_input), train_output, **self.loss_fn_kwargs) 
                loss.backward() 

                # compute auxiliary gradient for a larger batch size
                self.aux_model = copy.deepcopy(self.model)
                aux_loss = self.loss_fn(self.aux_model(aux_input), aux_output, **self.loss_fn_kwargs)
                aux_loss.backward()

                self._evoke_callback('on_after_backward')

                self.optimizer.step() 
                self.optimizer.zero_grad() 

                self.current_training_loss += loss.item()

                self._evoke_callback('on_train_batch_end')


            toc_epoch_end = time.time()
            self.current_epoch_training_time = toc_epoch_end - tic_epoch_start
            self.current_training_loss /= len(self.train_loader)

            self.epoch_metrics['training_loss'].append(self.current_training_loss)
            self.epoch_metrics['training_time'].append(self.current_epoch_training_time)
            self._evoke_callback('on_train_end')

            self.model.eval()
            self.current_val_loss = 0.0
            self.current_val_acc = 0.0
            for val_input, val_output in self.val_loader:
                
                self._evoke_callback('on_val_batch_start')
                
                # validation
                val_input, val_output = val_input.to(device), val_output.to(device)
                with torch.no_grad():
                    pred_output = self.model(val_input)

                val_loss = self.loss_fn(pred_output, val_output, **self.loss_fn_kwargs).item()
                self.current_val_loss += val_loss
                val_acc = accuracy(pred_output, val_output)
                self.current_val_acc += val_acc

                self._evoke_callback('on_val_batch_end')

            self.current_val_loss /= len(self.val_loader)
            self.current_val_acc /= len(self.val_loader)

            self.epoch_metrics['val_loss'].append(self.current_val_loss)
            self.epoch_metrics['val_acc'].append(self.current_val_acc.item())

            self._evoke_callback('on_val_end')

            # TODO: Probably don't need test set here; revisit.
            # train on training data --> tune on validation data --> final test on test data

            self._evoke_callback('on_epoch_end')

            if self.stop == True:
                break

            #print('epoch {} ended, metrics {}'.format(self.epoch, self.epoch_metrics))

        print('end training at epoch {}'.format(self.epoch))

        earlier_epoch = max(0, self.epoch - 10)
        self.epoch_metrics['val_acc_improvement'] = [self.epoch_metrics['val_acc'][-1] - self.epoch_metrics['val_acc'][earlier_epoch]]
        self.epoch_metrics['val_loss_improvement'] = [self.epoch_metrics['val_loss'][-1] - self.epoch_metrics['val_loss'][earlier_epoch]]

        return self.epoch_metrics


    def _evoke_callback(self, checkpoint_name, *args, **kwargs):
        for callback in self.callbacks:
            fn = getattr(callback, checkpoint_name)
            if callable(fn):
                fn(self, *args, **kwargs)