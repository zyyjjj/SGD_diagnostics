import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import SGD, Adagrad, Adam
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
import torchvision
import torchvision.transforms as transforms
import neptune.new as neptune
import os, sys, random
sys.path.append('../')
sys.path.append('/home/yz685/SGD_diagnostics/')
from src.utils.learner import Learner
from src.utils.callback import *

DEBUG = False
HPs_to_VARY = {
    'lr': ['uniform', [0.0001, 0.01]],
    'log2_batch_size': ['int', [5, 8]],
    'layer_1_size': ['int', [16, 64]],
    'layer_2_size': ['int',  [16, 64]],
    'dropout_rate': ['uniform', [0, 0.5]],
    'log2_aux_batch_size': ['int', [8, 10]],
    'iteration_fidelity': ['uniform', [0.25, 1]]
}

MultiFidelity_PARAMS = {
    "fidelity_dim": 6,
    "target_fidelities": {6: 1.0, 7: 0},
    "fidelity_weights": {6: 1},
    "fixed_cost": 1
}

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
max_num_epochs = 10 if DEBUG else 100

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

def problem_evaluate(X, return_metrics, designs = HPs_to_VARY, debug = DEBUG):
    """
    INPUTS:
    - X is a tensor of size (num_trials, num_hps_to_vary)
    - return_metrics is a dictionary {'metric_name': 'return_type'}
        where 'metric_name' in ['training_loss', 'val_loss', 'val_acc'] U {auxiliary epoch metrics}
        and 'return_type' in ['mean', 'last', 'max', 'min']
    OUTPUTS:
    - a tensor of shape num_trials x len(return_metrics)
    """

    input_shape = X.shape

    assert input_shape[1] == len(HPs_to_VARY), \
        'Input dimension 1 should match the number of hyperparameters to vary'

    outputs = []
    
    # TODO: think about whether we want to set the same seed as the BO trial here

    for i in range(input_shape[0]):
        base_config = {'lr': X[i][0].item(),
            'batch_size': 2**(X[i][1].item()),
            'aux_batch_size': 2**(X[i][5].item()),
            'running_avg_window_size': 10} # fix this parameter for now
        hp_config = [base_config, {}, {}]

        print('sampled config', X[i])

        model = MnistMlpModel(int(X[i][2].item()), int(X[i][3].item()), X[i][4].item())
        model.to(device)

        optimizer = torch.optim.Adam

        if debug:
            train_size = 4200
            val_size = 1800
            subds_indices = random.sample(range(len(train_ds_whole)), train_size + val_size)
            subds = Subset(train_ds_whole, subds_indices)
            train_ds, val_ds = random_split(subds, [train_size, val_size])
        else:   
            train_size, val_size = 42000, 18000 # TODO: don't hard code this         
            train_ds, val_ds = random_split(train_ds_whole, [train_size, val_size])

        # run = neptune.init(
        #     project="zyyjjj/SGD-diagnostics",
        #     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwM2RkNWY2MS01MTU1LTQxNDMtOTE3Ni1mN2RlNjY5ZTQxNWUifQ==",
        #     )
        # run["sys/tags"].add(['cifar'])  # organize things
        # run['params'] = hp_config + [{'arch_parmas': [X[i][3].item(), X[i][4].item()]}]
            
        loss_fn = torch.nn.functional.cross_entropy

        callbacks = [
            #EarlyStopping(metric = 'val_acc', patience = 5, warmup = 20, to_minimize=False, tolerance_thresh=0.05),
            AuxMetricsLogger()
        ]
    
        # construct learner and return a dictionary of {'metric': list of metric values over epochs}
        learner = Learner(hp_config, model, train_ds, val_ds, optimizer, loss_fn, callbacks)
        
        # X[i][-1] is iteration fidelity, i.e., the fraction of max_num_epochs to train for
        if 'iteration_fidelity' in designs:
            num_epochs = int(X[i][6] * max_num_epochs)
        else:
            num_epochs = max_num_epochs
            
        print('plan to train for {} epochs'.format(num_epochs))

        logged_performance_metrics = learner.fit(num_epochs, device = device)
        # print(logged_performance_metrics)

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
