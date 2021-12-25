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
from utils.nn_base import ImageClassificationBase

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

class MnistCnnModel(ImageClassificationBase):
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

    args, opt_kwargs, configs_set = parse_hp_args()
    print(opt_kwargs)
    optimizer = eval(args.optimizer)

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

    for config_pair in configs_set:

        print('Training on learning rate {} and batch size {}'.format(config_pair[0], config_pair[1]))

        run = neptune.init(
        project="zyyjjj/SGD-diagnostics",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwM2RkNWY2MS01MTU1LTQxNDMtOTE3Ni1mN2RlNjY5ZTQxNWUifQ==",
        )

        run["sys/tags"].add(['mnist'])  # organize things
        run['params'] = config_pair

        model = MnistCnnModel()
        model.to(device)
        loss_fn = torch.nn.functional.cross_entropy

        lr = config_pair[0]
        batch_size = config_pair[1]

        train_loader = DataLoader(train_ds, batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size)

        save_label = str(args.optimizer)+"_LR_"+str(lr)+"_" +\
                '_'.join('{}_{}'.format(*p) for p in sorted(opt_kwargs.items()))\
                + "_BATCH_"+str(batch_size)

        prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=10),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/'+save_label),
                profile_memory = True,
                record_shapes=True,
                with_stack=True)
        prof.start()

        fit(args.num_epochs, 
            lr, 
            model, 
            train_ds,
            args.aux_batch_size,
            train_loader, 
            val_loader, 
            trial = args.trial,
            loss_fn = loss_fn, 
            opt_func = optimizer, 
            save_label = save_label,
            device = device,
            run=run,
            prof = prof,
            opt_kwargs = opt_kwargs)



    