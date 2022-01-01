import torch
import torch.nn as nn
import torch.profiler
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
train_ds_whole = torchvision.datasets.MNIST('/home/yz685/SGD_diagnostics/experiments/mnist',
                                        train = True, download = True, transform = transforms.ToTensor())
test_ds = torchvision.datasets.MNIST('/home/yz685/SGD_diagnostics/experiments/mnist',
                                        train = False, download = True, transform = transforms.ToTensor())

print('train size {}, test size {}'.format(len(train_ds_whole), len(test_ds)))

train_frac = 0.9
train_size = int(len(train_ds_whole) * train_frac)
val_size = len(train_ds_whole) - train_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class MnistCnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 14 x 14

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 7 x 7

            nn.Flatten(), 
            nn.Linear(128*7*7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
        
    def forward(self, xb):
        return self.network(xb)


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
    
    train_ds, val_ds = random_split(train_ds_whole, [train_size, val_size])

    run = neptune.init(
        project="zyyjjj/SGD-diagnostics",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwM2RkNWY2MS01MTU1LTQxNDMtOTE3Ni1mN2RlNjY5ZTQxNWUifQ==",
        )

    run["sys/tags"].add(['mnist'])  # organize things
    run['params'] = hp_config
        
    model = MnistCnnModel()
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