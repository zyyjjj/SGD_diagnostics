import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

import numpy as np
import pdb, time, argparse, itertools, copy

dataset_size = 100000
train_frac = 0.7
train_size = int(dataset_size * train_frac)
val_size = dataset_size - train_size

# How to pass in a neural architecture? TODO: find example code
# FOR now: keep the simple architecture, train on different (lr, batch size) combos

def parse_hp_args():
    parser = argparse.ArgumentParser(description='input names and values of hyperparameters to vary')
    parser.add_argument('-o', '--optimizer', default = 'SGD', help='optimizer algorithm')
    parser.add_argument('-lr', '--learning_rates', nargs='+', action='append', type = float, help='learning rates to try')
    parser.add_argument('-b', '--batch_sizes', nargs='+', action='append', type = int, help = 'batch sizes to try', required=True)
    parser.add_argument('-ap', '--arch_params', action='append', 
            help='names of hyperparameters for the network architecture', required=True)
    parser.add_argument('-av', '--arch_param_values', required=True, action='append', type = int,
            help='values of hyperparameters for the network architecture')
    parser.add_argument('-T', '--num_epochs', default=10000, type=int, help='number of training-evaluation iterations to run')
    #parser.add_argument('-s', '--scheduling', action='store_true', help='run linear scheduling of learning rate')
    #parser.add_argument('--num_processes', default=8, type=int, help='number of processes to run for multiprocessing')

    args = parser.parse_args()

    #print('args.arch_params', args.arch_params)
    #print('args.arch_param_values', args.arch_param_values)
    print('batch sizes', args.batch_sizes[0])
    print('learning rates', args.learning_rates[0])
  
    if len(args.arch_params) != len(args.arch_param_values):
        raise(Exception('Number of architecture hyperparameters does not match number of values passed in'))

    configs_set = list(itertools.product(args.learning_rates[0], args.batch_sizes[0]))
    print('configs_set {}'.format(configs_set))

    arch_param_dict = dict(zip(args.arch_params, args.arch_param_values))
    print(arch_param_dict)

    # TODO: return dictionary of architecture parameter values
    return args, arch_param_dict, configs_set

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 2. train 

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


if __name__ == "__main__":

    args, arch_param_dict, configs_set = parse_hp_args()
    input_dim = arch_param_dict['input_dim']
    middle_dim = arch_param_dict['middle_dim']

    inputs = torch.rand((dataset_size, input_dim))
    targets = torch.reshape(torch.norm(inputs, dim=1), (dataset_size, 1))
    inputs = inputs.to(device)
    targets = targets.to(device)

    dataset = TensorDataset(inputs, targets)
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    model = nn.Sequential(
        nn.Linear(input_dim, middle_dim),
        nn.ReLU(),
        nn.Linear(middle_dim, 1)
    )
    model.to(device)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    writer = SummaryWriter()

    # TODO: implement multiprocessing

    for config_pair in configs_set:
        train_loader = DataLoader(train_ds, config_pair[1], shuffle=True)
        val_loader = DataLoader(val_ds, config_pair[1])
        fit(args.num_epochs, config_pair[0], model, train_loader, val_loader)

    