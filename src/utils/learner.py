import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
import random, time, copy, os

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

        self.optimizer = opt(self.model.parameters(), self.base_config['lr'], **self.opt_kwargs)
        self.loss_fn = loss_fn(**self.loss_fn_kwargs)
        self.stop = False

        self.logged_metrics = None # TODO: or empty dictionary?

    def fit(self, num_epochs, trial, save_label, device, run):

        # TODO: move the code from train_nn.py to here

        script_dir = os.path.abspath(os.getcwd())
        # TODO: may need to update logging structure 
        # results_folder = script_dir + "/results/" + save_label + "trial_" + str(trial) + "/" 
        results_folder = script_dir + "/results/" + save_label + "/" 
        os.makedirs(results_folder, exist_ok=True)

        # TODO: HERE initialize empty tensors and lists here
        # and create directories for quantities to record
        # have MetricsLogger do this

        for epoch in range(num_epochs):

            self.model.train()
            self._evoke_callback('on_epoch_start')

            tic_epoch_start = time.time()

            aux_indices = random.sample(range(len(self.train_ds)), self.aux_batch_size)
            auxiliary_ds = Subset(self.train_ds, aux_indices) 
            aux_loader = DataLoader(auxiliary_ds, len(auxiliary_ds))
            for aux_data in aux_loader:
                aux_input, aux_output = aux_data
            aux_input = aux_input.to(device)
            aux_output = aux_output.to(device)
        
            batch_counter = 0

            accum_grad = None

            training_loss = 0.0

            for train_input, train_output in self.train_loader:
                self._evoke_callback('on_train_batch_start')

                # train and log
                train_input, train_output = train_input.to(device), train_output.to(device)
                loss = self.loss_fn(self.model(train_input), train_output) 
                loss.backward() 

                # compute auxiliary gradient for a larger batch size
                self.aux_model = copy.deepcopy(self.model)
                aux_loss = self.loss_fn(self.aux_model(aux_input), aux_output)
                aux_loss.backward()

                self._evoke_callback('on_after_backward')

                self.optimizer.step() 
                self.optimizer.zero_grad() 

                training_loss += loss.item()

                self._evoke_callback('on_train_batch_end')
                # TODO: HERE 1. log batch gradient and aux batch gradient
                # 2. also compute norm of the two
                # 3. accumulate the epoch gradient
                # 4. compute cosine sim, norm of change, and other signals
                # 5. compute running avg of gradients / squared norm of gradients
                # 6. monitor memory
                # append to list and log in neptune run

            toc_epoch_end = time.time()
            training_loss /= len(self.train_loader)
            self._evoke_callback('on_train_end')
            # TODO: HERE log training time, training loss, accum_grad, model_params
            # save the batch-wise metric values in this epoch

            # with saving metrics: play with regular expressions
            # make them both a function and folder name

            
            self.model.eval()
            validation_loss = 0.0
            validation_acc = 0.0
            for val_input, val_output in self.val_loader:
                self._evoke_callback('on_val_batch_start')
                
                # validation
                val_input, val_output = val_input.to(device), val_output.to(device)
                with torch.no_grad():
                    pred_output = self.model(val_input)

                val_loss = self.loss_fn(pred_output, val_output).item()
                validation_loss += val_loss

                val_acc = accuracy(pred_output, val_output)
                validation_acc += val_acc

                self._evoke_callback('on_val_batch_end')

            validation_loss /= len(self.val_loader)
            validation_acc /= len(self.val_loader)

            self._evoke_callback('on_val_end')
            # TODO: HERE log val_loss and val_acc
            # also keep a history of val_loss for early stopping


            # TODO: Test set?????
            # should be ok, not needed here
            # train on training data --> tune on validation data --> final test on test data

            self._evoke_callback('on_epoch_end')
            # TODO HERE save the model

            if self.stop == True:
                break


    def _evoke_callback(self, checkpoint_name, *args, **kwargs):
        for callback in self.callbacks:
            fn = getattr(callback, checkpoint_name)
            if callable(fn):
                fn(self, *args, **kwargs)
    

    # TODO: write functions for different custom signals 
    def compute_metric():
        pass