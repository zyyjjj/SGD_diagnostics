import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.profiler
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adagrad, Adam
import torchvision
import torchvision.transforms as transforms
import neptune.new as neptune
import random
import numpy as np
import pdb, time, argparse, itertools, copy
import sys, os
from utils.parse_hp_args import parse_hp_args
from utils.train_nn import fit, accuracy
from utils.learner import Learner
from utils.callback import *

# TODO: eventually move these codes to experiments / examples folder 

# 60K
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_ds_whole = torchvision.datasets.CIFAR10('/home/yz685/SGD_diagnostics/experiments/cifar',
                                        train = True, download = True, transform = transform)
test_ds = torchvision.datasets.CIFAR10('/home/yz685/SGD_diagnostics/experiments/cifar',
                                        train = False, download = True, transform = transform)

print('train size {}, test size {}'.format(len(train_ds_whole), len(test_ds)))

train_frac = 0.9
train_size = int(len(train_ds_whole) * train_frac)
val_size = len(train_ds_whole) - train_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class CifarCnnModel(nn.Module):
    # TODO: find a better (SOTA) network architecture

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


    def __init__(self, n_channels_1 = 16, n_channels_2 = 16):

        super().__init__()
        
        # TODO: enable customizing the architecture from input

        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, n_channels_1, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(n_channels_1, n_channels_1, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2) # output is n_channels_1 * 16 * 16
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(n_channels_2, n_channels_2, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(n_channels_2, n_channels_2, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2) # output is n_channels_2 * 8 * 8
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(4 * 4 * n_channels_2, 256),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(256, 10)

        self.softmax = nn.Softmax()
        

    def forward(self, x):

        out = self.convblock1(x)
        out = self.convblock2(out)
        out = self.convblock2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        
        return out


if __name__ == "__main__":

    args, base_config, opt_kwargs, loss_fn_kwargs = parse_hp_args()
    optimizer = eval(args.optimizer)
    hp_config = [base_config, opt_kwargs, loss_fn_kwargs]

    print('Training with the following config', hp_config)

    seed = args.trial
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # TODO: figure out test vs. validation data
    train_ds, val_ds = random_split(train_ds_whole, [train_size, val_size])

    run = neptune.init(
        project="zyyjjj/SGD-diagnostics",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwM2RkNWY2MS01MTU1LTQxNDMtOTE3Ni1mN2RlNjY5ZTQxNWUifQ==",
        )

    run["sys/tags"].add(['cifar'])  # organize things
    run['params'] = hp_config
        
    model = CifarCnnModel()
    print(model)
    model.to(device)
    # TODO: rather than cross entropy, can directly compute classification error
    loss_fn = torch.nn.functional.cross_entropy
    
    save_label = str(args.optimizer) + '_' + \
            '_'.join('{}_{}'.format(*p) for p in sorted(base_config.items())) + \
            '_'.join('{}_{}'.format(*p) for p in sorted(opt_kwargs.items())) + \
            '_'.join('{}_{}'.format(*p) for p in sorted(loss_fn_kwargs.items()))

    script_dir = os.path.abspath(os.getcwd())
    results_folder = script_dir + "/results/" + save_label + "/" + "trial_" + str(args.trial) + "/" 
    os.makedirs(results_folder, exist_ok=True)

    callbacks = [MetricsLogger(results_folder), EarlyStopping(metric = 'val_acc', patience = 5, warmup = 20, to_minimize=False, tolerance_thresh=0.05)]

    learner = Learner(model, train_ds, val_ds, optimizer, loss_fn, hp_config, callbacks, run)

    learner.fit(args.num_epochs, device = device)



# TODO: don't forget the ReLU activation statistics idea
# or more broadly, which parts of the connections light up most frequently?
# can write a custom callback class to monitor this