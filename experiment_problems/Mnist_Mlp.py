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
    'layer_1_size': ['int', [16, 64]],
    'layer_2_size': ['int',  [16, 64]],
    'dropout_rate': ['uniform', [0, 0.5]],
    'log2_aux_batch_size': ['int', [5, 10]]
}

# TODO: we still want aux batch size for this (this is indep of type of network)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_frac = 0.7
train_ds_whole = torchvision.datasets.MNIST('/home/yz685/SGD_diagnostics/experiments/mnist',
                                train = True, download = True, transform = transforms.ToTensor())
test_ds = torchvision.datasets.MNIST('/home/yz685/SGD_diagnostics/experiments/mnist',
                                train = False, download = True, transform = transforms.ToTensor())

train_size = int(len(train_ds_whole) * train_frac)  
val_size = len(train_ds_whole) - train_size
max_num_epochs = 100

class MnistMlpModel(nn.Module):
    # A proof-of-concept experiment, using a 2-layer MLP with dropout on MNIST
    def __init__(self, layer_1_size, layer_2_size, dropout_rate):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, layer_1_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(layer_1_size, layer_2_size),
            nn.Dropout(dropout_rate),
            nn.Linear(layer_2_size, 10),
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
            'aux_batch_size': 2**(X[i][5].item())}
        hp_config = [base_config, {}, {}]

        print('sampled config', X[i])

        model = MnistMlpModel(int(X[i][2].item()), int(X[i][3].item()), X[i][4].item())
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
