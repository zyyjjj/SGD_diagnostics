import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]
class Learner():
    def __init__(self, model, train_ds, val_ds, opt, loss_fn, hp_config, callbacks):

        self.model, self.train_ds, self.val_ds = model, train_ds, val_ds
        
        self.callbacks = listify(callbacks)

        self.base_config, self.opt_kwargs, self.loss_fn_kwargs = hp_config
        
        self.train_loader = DataLoader(train_ds, self.base_config['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_ds, self.base_config['batch_size'])

        # TODO: can you automate this? or is this needed?
        #self.aux_batch_size = self.base_config['aux_batch_size']
        #self.lr = self.base_config['lr']
        #self.running_avg_window_size = self.base_config['running_avg_window_size']

        self.optimizer = opt(self.model.parameters(), self.lr, **self.opt_kwargs)
        self.loss_fn = loss_fn(**self.loss_fn_kwargs)
        self.stop = False

        #have some version of this to enable early stopping
        self.metric_history = None

    def fit(self, num_epochs, trial, save_label, device, callbacks, run):

        # TODO: move the code from train_nn.py to here

        for epoch in range(num_epochs):

            self.model.train()
            self._evoke_callback('on_epoch_start')


            for train_input, train_output in self.train_loader:
                self._evoke_callback('on_train_batch_start')

                # train and log


                self._evoke_callback('on_train_batch_end')

            
            self.model.eval()
            for val_input, val_output in self.val_loader:
                self._evoke_callback('on_val_batch_start')

                # validation

                self._evoke_callback('on_val_batch_end')


            # Test set?????

            self._evoke_callback('on_epoch_end')

            if self.stop == True:
                break

        pass



    def _evoke_callback(self, checkpoint_name, *args, **kwargs):
        for callback in self.callbacks:
            fn = getattr(callback, checkpoint_name)
            if callable(fn):
                fn(self, *args, **kwargs)