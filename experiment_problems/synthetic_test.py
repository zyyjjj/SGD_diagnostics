import torch
import sys, os
sys.path.append('../')
sys.path.append('/home/yz685/SGD_diagnostics/')

Input_RANGES = {
    'x': ['uniform', [1.0, 5.0]],
    'iteration_fidelity': ['uniform', [0.25, 1]]
}

MultiFidelity_PARAMS = {
    "fidelity_dim": 1,
    "target_fidelities": {1: 1.0},
    "fidelity_weights": {1: 1},
    "fixed_cost": 0
}

checkpoint_fidelities = [0.25, 0.5, 1]


def problem_evaluate(X, is_multitask, checkpoint_fidelities = checkpoint_fidelities):
    # input (x, s)
    # output task(s), where task 0 = [noise, noise, ..., x], task 1 = [x, x, ..., x]

    input_shape = X.shape
    outputs = []
    max_iters = 20

    for i in range(input_shape[0]):
        x = X[i][0].item()
        s = X[i][1].item()
        y = 10 - (x-3)**2

        checkpoints = []
        for frac_fid in checkpoint_fidelities:
            checkpoints.append(int(max_iters * s * frac_fid))

        for checkpoint in checkpoints:
            print('evaluating at checkpoint {}'.format(checkpoint))
            if checkpoint < max_iters - 1:
                task_0 = torch.randn(1)
            else:
                task_0 = torch.tensor([y])
            task_1 = torch.tensor([y])

            if is_multitask:
                outputs.append(torch.cat((task_0, task_1)))
            else:
                outputs.append(task_0)

    print(outputs)    
    return torch.stack(outputs)


"""
March 14 observations
1. samples at fidelity = 1 at a few initial iterations, then keep sampling at 0.25
2. when sampling at fidelity = 1, acquisition function value is negative? 
   but when sampling at 0.25, acquisition function value becomes positive again ... interesting
3. think about how to effectively compare STMF and MTMF in this scenario
"""