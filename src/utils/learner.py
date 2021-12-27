import numpy as np
import torch
from torch.utils.data import DataLoader, random_split




class Learner():
    def __init__(self, model, train_ds, val_ds, opt, loss_fn, hp_config, callback=None):
        self.model, self.train_ds, self.val_ds = model, train_ds, val_ds
        
        # TODO: listify
        # if callback: self.callback = callback

        self.base_config, self.opt_kwargs, self.loss_fn_kwargs = hp_config
        
        self.batch_size = self.base_config['batch_size']
        self.train_loader = DataLoader(train_ds, self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, self.batch_size)

        self.aux_batch_size = self.base_config['aux_batch_size']
        self.lr = self.base_config['lr']

        self.optimizer = opt(self.model.parameters(), self.lr, **self.opt_kwargs)

        self.loss_fn = loss_fn(**self.loss_fn_kwargs)