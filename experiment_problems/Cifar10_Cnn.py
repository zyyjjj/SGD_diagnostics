import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import SGD, Adagrad, Adam
import torchvision
import torchvision.transforms as transforms
import neptune.new as neptune
import os, sys, random

sys.path.append('../')
sys.path.append('/home/yz685/SGD_diagnostics/')
from src.utils.learner import Learner
from src.utils.callback import *


HPs_to_VARY = {
    'lr': ['uniform', [-5, -2]],
    'log2_batch_size': ['int', [5, 10]],
    'log2_aux_batch_size': ['int', [5, 10]],
    'n_channels_1': ['int', [32, 64]],
    'n_channels_2': ['int',  [32, 64]],
    'iteration_fidelity': ['uniform', [0.25, 1]]
}

MultiFidelity_PARAMS = {
    "fidelity_dim": 5,
    "target_fidelities": {5: 1.0}, # {5: 1.0, 6: 0},
    "fidelity_weights": {5: 1},
    "fixed_cost": 1
}

checkpoint_fidelities = [0.25, 0.5, 1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    # transforms.RandomHorizontalFlip(0.4),
    # transforms.RandomAffine(0, translate  = (0.25, 0.25))
    ])

DEBUG = True

train_frac = 0.8
train_ds_whole = torchvision.datasets.CIFAR10('/home/yz685/SGD_diagnostics/experiments/cifar',
                                train = True, download = True, transform = transform)
test_ds = torchvision.datasets.CIFAR10('/home/yz685/SGD_diagnostics/experiments/cifar',
                                train = False, download = True, transform = transform)

train_size = int(len(train_ds_whole) * train_frac)
val_size = len(train_ds_whole) - train_size

max_num_epochs = 20


class CifarCnnModel(nn.Module):
    """
    Experiments in Jian's paper:
    1. Simple proof-of-concept on MNIST
        1) architecture: 2-layer MLP
        2) hps: lr, dropout rate, batch size, number of units in layer 1 and layer 2
    2. CNN on CIFAR-10 and SVHN
        1) architecture: 3 conv blocks and a soft max classification layer; each block has two conv layers with the same # filters, followed by a max-pooling layer; no dropout or batchnorm
            ("Filter" is the same as "out_channels" in torch.nn.Conv2d())
        2) standard data augmentation: horizontal and vertical shifts, horizontal flips
        3) hps: lr, batch size, number of filters in conv blocks 1, 2, and 3
    """
    def __init__(self, n_channels_1, n_channels_2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = n_channels_1, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = n_channels_1, out_channels = n_channels_1, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # output is n_channels_1 * 16 * 16

            nn.Conv2d(in_channels = n_channels_1, out_channels = n_channels_2, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = n_channels_2, out_channels = n_channels_2, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # output is n_channels_2 * 8 * 8

            nn.Conv2d(in_channels = n_channels_2, out_channels = n_channels_2, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = n_channels_2, out_channels = n_channels_2, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # output is n_channels_2 * 4 * 4

            nn.Flatten(),
            nn.Linear(4 * 4 * n_channels_2, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Softmax()
        )

    def forward(self, x):
        return self.network(x)

def problem_evaluate(X, return_metrics, designs = HPs_to_VARY, checkpoint_fidelities = checkpoint_fidelities, debug = DEBUG):
    """
    INPUTS:
    X: a tensor of size (num_trials, num_hps_to_vary)
    return_metrics: a dictionary {'metric_name': 'return_type'}
        where 'metric_name' in ['training_loss', 'val_loss', 'val_acc']
        and 'return_type' in ['mean', 'last', 'max', 'min']
    checkpoint_fidelities: a list of numbers <= 1, with the last entry being 1, 
        representing the checkpoints of iteration fidelity values at which we want to log outputs
    """

    input_shape = X.shape

    assert input_shape[1] == len(HPs_to_VARY), \
        'Input dimension 1 should match the number of hyperparameters to vary'

    outputs = [] # make this multi-output
    
    # TODO: think about whether we want to set the same seed as the BO trial here

    for i in range(input_shape[0]):

        base_config = {'lr': 10**(X[i][0].item()),
            'batch_size': 2**(X[i][1].item()),
            'aux_batch_size': 2**(X[i][2].item()),
            'running_avg_window_size': 10}        
        hp_config = [base_config, {}, {}]

        print('sampled config', X[i])

        model = CifarCnnModel(int(X[i][3].item()), int(X[i][4].item()))
        model.to(device)

        # pdb.set_trace()

        # optimizer = torch.optim.Adam
        optimizer = torch.optim.SGD

        if debug:
            train_size = 4000
            val_size = 1000
            subds_indices = random.sample(range(len(train_ds_whole)), train_size + val_size)
            subds = Subset(train_ds_whole, subds_indices)
            train_ds, val_ds = random_split(subds, [train_size, val_size])
        else:   
            train_size, val_size = 40000, 10000 # TODO: don't hard code this         
            train_ds, val_ds = random_split(train_ds_whole, [train_size, val_size])

        # run = neptune.init(
        #     project="zyyjjj/SGD-diagnostics",
        #     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwM2RkNWY2MS01MTU1LTQxNDMtOTE3Ni1mN2RlNjY5ZTQxNWUifQ==",
        #     )
        # run["sys/tags"].add(['cifar_BO'])  # organize things
        # run['params'] = hp_config + [{'arch_parmas': [X[i][3].item(), X[i][4].item()]}]
            
        loss_fn = torch.nn.CrossEntropyLoss()
        callbacks = [
            #EarlyStopping(metric = 'val_acc', patience = 5, warmup = 20, to_minimize=False, tolerance_thresh=0.05),
            AuxMetricsLogger()
        ]
        # learner at 0x7f674709c610
        # learner.model.parameters() 0x7f66e044fac0 -- this changes every round 0x7f66e0345f20 
        # learner.model.parameters() 0x7f1a465d8ac0
        # self.model.parameters()  0x7f1a465d8a50
        # but loss is increasing

        learner = Learner(hp_config, model, train_ds, val_ds, optimizer, loss_fn, callbacks)
        
        if 'iteration_fidelity' in designs:
            num_epochs = int(X[i][5] * max_num_epochs)
        else:
            num_epochs = max_num_epochs
    
        # list of checkpoints (a few fidelities) where outputs are logged
        checkpoint_epochs = []
        for frac_fid in checkpoint_fidelities:
            checkpoint_epochs.append(min(int(X[i][5] * frac_fid * max_num_epochs)+1, num_epochs) )
        
        print('plan to train for {} epochs'.format(num_epochs))
        print('checkpoint epochs: ', checkpoint_epochs)

        logged_performance_metrics = learner.fit(num_epochs, device = device)
        
        # then, extract intermediate signals from logged_perf_metrics[n_epochs] 
        # for n_epochs in checkpoint_epochs
        # desired output y shape: (n_trials * n_checkpoints) x num_outputs, e.g.
        # i.e., rows are in the order of (trial 0, fid 0, all outputs), (trial 0, fid 1, all outputs), ...
        
        for checkpoint in checkpoint_epochs:
            print('extracting outputs at checkpoint {}'.format(checkpoint))
            output = []
            for k,t in return_metrics.items():
                if t == 'mean':
                    out = np.mean(logged_performance_metrics[k][:checkpoint])
                elif t == 'last':
                    print('key: ', k)
                    print(logged_performance_metrics[k])
                    # print('length of stored metric history: ', len(logged_performance_metrics[k]) )
                    out = logged_performance_metrics[k][checkpoint-1] 
                    # TODO: there's a bug here?
                    # likely because I only logged one number rather than the full history for some metrics
                elif t == 'max':
                    out = max(logged_performance_metrics[k][:checkpoint])
                elif t == 'min':
                    out = min(logged_performance_metrics[k][:checkpoint])
            
                output.append(out)

            outputs.append(output)
    
    # return a tensor of shape (num_trials * num_checkpoints) x len(return_metrics)
    return torch.Tensor(outputs)