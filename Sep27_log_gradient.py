import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

import numpy as np
import pdb
import time

import copy

input_dim = 3
middle_dim = 2
dataset_size = 100000
train_frac = 0.7
train_size = int(dataset_size * train_frac)
val_size = dataset_size - train_size
batch_size = 128
lr = 1e-5
num_epochs = 20000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 1. generate dataset and split into training and validation

inputs = torch.rand((dataset_size, input_dim))
targets = torch.reshape(torch.norm(inputs, dim=1), (dataset_size, 1))

inputs = inputs.to(device)
targets = targets.to(device)

print('are inputs and targets on the GPU?', inputs.is_cuda, targets.is_cuda)

# or: 
# inputs = torch.rand((dataset_size, input_dim), device = device)
# targets = torch.reshape(torch.norm(inputs, dim=1), (dataset_size, 1), device = device)

dataset = TensorDataset(inputs, targets)
train_ds, val_ds = random_split(dataset, [train_size, val_size])
#print('Are training and validation data sets on the GPU? ', train_ds.is_cuda, val_ds.is_cuda)

model = nn.Sequential(
    nn.Linear(input_dim, middle_dim),
    nn.ReLU(),
    nn.Linear(middle_dim, 1)
)
model.to(device)
loss_fn = torch.nn.MSELoss(reduction='mean')
print('use batch-mean MSE as loss function')

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

writer = SummaryWriter()

# 2. train without adding noise to gradient

"""


 with torch.profiler.profile(
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=6,
        repeat=1),
    on_trace_ready=tensorboard_trace_handler,
    with_stack=True
) as profiler:
"""



def fit(num_epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):

    start_time = time.time()

    optimizer = opt_func(model.parameters(), lr)

    #scheduler = ExponentialLR(optimizer, gamma=0.9)

    num_params = len(optimizer.param_groups[0]['params'])
    prev_param_values = copy.deepcopy(optimizer.param_groups[0]['params'])
    prev_grad = [None] * num_params
    prev_concat_grad = None

    # TODO: learning rate scheduling https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    # OR momentum-based SGD

    for epoch in range(num_epochs):

        if epoch % 20 == 0:
            print('training epoch {}'.format(epoch))

        batch_counter = 0

        accum_grad = [None] * num_params
        
        # Training Phase 
        training_loss = 0.0
        for batch in train_loader:
            batch_counter += 1

            train_input, train_output = batch
            loss = loss_fn(model(train_input), train_output) # generate predictions and calculate loss
            loss.backward() # compute gradients

            for p in range(num_params):
                gradient = optimizer.param_groups[0]['params'][p].grad

                if accum_grad[p] is None:
                    accum_grad[p] = copy.deepcopy(gradient)
                else:
                    accum_grad[p] = copy.deepcopy(accum_grad[p] + gradient)
            
            optimizer.step() # update weights
            optimizer.zero_grad() # reset gradients to zero

            training_loss += loss.item()

        #scheduler.step()

        accum_grad = [accum_grad[p]/batch_counter for p in range(len(accum_grad))]
        training_loss /= len(train_loader)

        writer.add_scalar('training loss', training_loss, epoch)

        # log norm of concatenated gradient
        accum_grad_flatten=[torch.flatten(accum_grad[p]) for p in range(len(accum_grad))]
        concat_grad = torch.cat(tuple(accum_grad_flatten))
        writer.add_scalar('norm of concatenated gradient', torch.linalg.norm(concat_grad), epoch)

        if prev_concat_grad is not None:
            writer.add_scalar('cosine similarity between previous and current concatenated gradients', \
                F.cosine_similarity(prev_concat_grad, concat_grad, dim=0), epoch)
            norm_concat_grad_change = torch.linalg.norm(concat_grad - prev_concat_grad)
            writer.add_scalar('norm of change in concatenated gradient', norm_concat_grad_change, epoch)

        prev_concat_grad = copy.deepcopy(concat_grad)

        # log norm of individual parameter's gradient
        for p in range(num_params):
            if prev_grad[p] is None:
                norm_grad_change = torch.linalg.norm(accum_grad[p])
            else:
                norm_grad_change = torch.linalg.norm(prev_grad[p] - accum_grad[p])
            
            norm_param_change = torch.linalg.norm(prev_param_values[p] - optimizer.param_groups[0]['params'][p])

            writer.add_scalar('norm of gradient change in param '+str(p), norm_grad_change, epoch)
            writer.add_scalar('norm of change in param '+str(p), norm_param_change, epoch)

        prev_grad = copy.deepcopy(accum_grad)
        prev_param_values = copy.deepcopy(optimizer.param_groups[0]['params'])

        # Validation phase
        validation_loss = 0.0
        for val_batch in val_loader:
            val_input, val_output = val_batch
            val_loss = loss_fn(model(val_input), val_output).item()
            validation_loss += val_loss

        validation_loss /= len(val_loader)
        writer.add_scalar('avg validation loss', validation_loss, epoch)
    
    print('finished training {} epochs, took {} seconds'.format(num_epochs, time.time()-start_time))


# 3. train with manual noise added to gradient
# 4. log:
#  4.1 x.grad
#  4.2 change in direction of x.grad --> what metric to use? cosine similarity?
#  4.3 change in model.params
#  4.4 training loss
#  4.5 step size
#  4.6 batch size



if __name__ == "__main__":
    fit(num_epochs, lr, model, train_loader, val_loader)
