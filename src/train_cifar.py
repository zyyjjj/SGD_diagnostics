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
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":

    args, base_config, opt_kwargs, loss_fn_kwargs = parse_hp_args()
    optimizer = eval(args.optimizer)
    hp_config = [base_config, opt_kwargs, loss_fn_kwargs]

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
    model.to(device)
    loss_fn = torch.nn.functional.cross_entropy
    callbacks = []

    learner = Learner(model, train_ds, val_ds, optimizer, loss_fn, hp_config, callbacks)

    save_label = str(args.optimizer) + \
            '_'.join('{}_{}'.format(*p) for p in sorted(base_config.items())) + \
            '_'.join('{}_{}'.format(*p) for p in sorted(opt_kwargs.items())) + \
            '_'.join('{}_{}'.format(*p) for p in sorted(loss_fn_kwargs.items()))


    learner.fit(args.num_epochs, 
        trial = args.trial,
        save_label = save_label,
        device = device,
        run=run)



# TODO: don't forget the ReLU activation statistics idea
# or more broadly, which parts of the connections light up most frequently?
# can write a custom callback class to monitor this