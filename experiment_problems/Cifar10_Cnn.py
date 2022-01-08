import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adagrad, Adam
import torchvision
import torchvision.transforms as transforms
import neptune.new as neptune
import os, sys
sys.path.append('../')
sys.path.append('/home/yz685/SGD_diagnostics/')
from src.utils.learner import Learner
from src.utils.callback import *


HPs_to_VARY = {
    'lr': ['uniform', [0.0001, 0.01]],
    'log2_batch_size': ['int', [5, 10]],
    'log2_aux_batch_size': ['int', [5, 10]],
    'n_channels_1': ['int', [16, 64]],
    'n_channels_2': ['int',  [16, 64]]
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_frac = 0.7

train_ds_whole = torchvision.datasets.CIFAR10('/home/yz685/SGD_diagnostics/experiments/cifar',
                                train = True, download = True, transform = transform)
test_ds = torchvision.datasets.CIFAR10('/home/yz685/SGD_diagnostics/experiments/cifar',
                                train = False, download = True, transform = transform)

train_size = int(len(train_ds_whole) * train_frac)
val_size = len(train_ds_whole) - train_size

max_num_epochs = 300


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

def problem_evaluate(X, return_metrics):
    """
    X is a tensor of size (num_trials, num_hps_to_vary)
    return_metrics is a dictionary {'metric_name': 'return_type'}
        where 'metric_name' in ['training_loss', 'val_loss', 'val_acc']
        and 'return_type' in ['mean', 'last', 'max', 'min']
    """

    input_shape = X.shape

    #print(input_shape, len(HPs_to_VARY))

    assert input_shape[1] == len(HPs_to_VARY), \
        'Input dimension 1 should match the number of hyperparameters to vary'

    outputs = []
    
    # TODO: think about whether we want to set the same seed as the BO trial here

    for i in range(input_shape[0]):
        base_config = {'lr': X[i][0].item(),
            'batch_size': 2**(X[i][1].item()),
            'aux_batch_size': 2**(X[i][2].item())}        
        hp_config = [base_config, {}, {}]

        print('sampled config', X[i])

        model = CifarCnnModel(int(X[i][3].item()), int(X[i][4].item()))
        model.to(device)

        optimizer = torch.optim.Adam
        train_ds, val_ds = random_split(train_ds_whole, [train_size, val_size])

        # run = neptune.init(
        #     project="zyyjjj/SGD-diagnostics",
        #     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwM2RkNWY2MS01MTU1LTQxNDMtOTE3Ni1mN2RlNjY5ZTQxNWUifQ==",
        #     )
        # run["sys/tags"].add(['cifar'])  # organize things
        # run['params'] = hp_config + [{'arch_parmas': [X[i][3].item(), X[i][4].item()]}]
            
        loss_fn = torch.nn.functional.cross_entropy
        callbacks = [EarlyStopping(metric = 'val_acc', patience = 5, warmup = 20, to_minimize=False, tolerance_thresh=0.05)]

        learner = Learner(hp_config, model, train_ds, val_ds, optimizer, loss_fn, callbacks)
        logged_performance_metrics = learner.fit(max_num_epochs, device = device)

        output = []
        for k,t in return_metrics.items():
            if t == 'mean':
                out = np.mean(logged_performance_metrics[k])
            elif t == 'last':
                out = logged_performance_metrics[k][-1]
            elif t == 'max':
                out = max(logged_performance_metrics[k])
            elif t == 'min':
                out = min(logged_performance_metrics[k])
        
            output.append(out)

        outputs.append(output)
    
    # return a tensor of shape num_trials x len(return_metrics)
    return torch.Tensor(outputs)