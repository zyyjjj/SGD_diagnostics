import torch.nn as nn
from utils.learner import Learner
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adagrad, Adam
import torchvision
import torchvision.transforms as transforms
import neptune.new as neptune
from utils.callback import *
import os

# TODO: write problem_evaluate for MNIST
class MnistCnnModel(nn.Module):
    # TODO: allow customizing architecture, maybe include dropout rate, or change to MLP (simpler)
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




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def problem_evaluate(hp_config, problem_name, n_trials=1):
    
    train_frac = 0.9
    
    # maybe pass in hp_config as a tuple
    # then unpack it 
    base_config, arch_config, opt_kwargs, loss_fn_kwargs = hp_config

    if problem_name == 'CIFAR-10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_ds_whole = torchvision.datasets.CIFAR10('/home/yz685/SGD_diagnostics/experiments/cifar',
                                        train = True, download = True, transform = transform)
        test_ds = torchvision.datasets.CIFAR10('/home/yz685/SGD_diagnostics/experiments/cifar',
                                        train = False, download = True, transform = transform)

        train_size = int(len(train_ds_whole) * train_frac)
        val_size = len(train_ds_whole) - train_size

        train_ds, val_ds = random_split(train_ds_whole, [train_size, val_size])

        run = neptune.init(
            project="zyyjjj/SGD-diagnostics",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwM2RkNWY2MS01MTU1LTQxNDMtOTE3Ni1mN2RlNjY5ZTQxNWUifQ==",
            )

        run["sys/tags"].add(['cifar'])  # organize things
        run['params'] = hp_config
            
        model = CifarCnnModel(arch_config)
        model.to(device)
        loss_fn = torch.nn.functional.cross_entropy()

    elif problem_name == 'MNIST':
        train_ds_whole = torchvision.datasets.MNIST('/home/yz685/SGD_diagnostics/experiments/mnist',
                                        train = True, download = True, transform = transforms.ToTensor())
        test_ds = torchvision.datasets.MNIST('/home/yz685/SGD_diagnostics/experiments/mnist',
                                        train = False, download = True, transform = transforms.ToTensor())

        train_size = int(len(train_ds_whole) * train_frac)  
        val_size = len(train_ds_whole) - train_size

        train_ds, val_ds = random_split(train_ds_whole, [train_size, val_size])

        run = neptune.init(
            project="zyyjjj/SGD-diagnostics",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwM2RkNWY2MS01MTU1LTQxNDMtOTE3Ni1mN2RlNjY5ZTQxNWUifQ==",
            )

        run["sys/tags"].add(['cifar'])  # organize things
        run['params'] = hp_config
            
        model = MnistCnnModel(arch_config)
        model.to(device)
        loss_fn = torch.nn.functional.cross_entropy()

    #str(args.optimizer)?
    save_label = 'Adam' + \ 
    '_'.join('{}_{}'.format(*p) for p in sorted(base_config.items())) + \
    '_'.join('{}_{}'.format(*p) for p in sorted(arch_config.items())) + \
    '_'.join('{}_{}'.format(*p) for p in sorted(opt_kwargs.items())) + \
    '_'.join('{}_{}'.format(*p) for p in sorted(loss_fn_kwargs.items()))

    script_dir = os.path.abspath(os.getcwd())
    results_folder = script_dir + "/results/" + save_label + "/" + "trial_" + str(args.trial) + "/" 
    os.makedirs(results_folder, exist_ok=True)

    callbacks = [MetricsLogger(results_folder), EarlyStopping(metric = 'val_acc', patience = 5, warmup = 20, to_minimize=False, tolerance_thresh=0.05)]

    # TODO: specify optimizer
    learner = Learner(hp_config, model, train_ds, val_ds, optimizer, loss_fn, callbacks, run)

    # TODO: call learner.fit() and return metric(s) of interest