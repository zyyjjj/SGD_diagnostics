import time, sys, os, random, copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import parameters_to_vector
import torch.nn.functional as F
import pdb
from .get_memory_info import get_memory_info 
from torch.profiler import profile, record_function, ProfilerActivity


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def fit(num_epochs, 
        lr, 
        model, 
        train_ds, 
        aux_batch_size,
        train_loader, 
        val_loader, 
        trial,
        loss_fn, 
        opt_func, 
        save_label,
        device,
        run,
        prof, 
        opt_kwargs={},
        window_size = 10):

    start_time = time.time()

    script_dir = os.path.abspath(os.getcwd())
    # TODO: may need to update logging structure 
    # results_folder = script_dir + "/results/" + save_label + "trial_" + str(trial) + "/" 
    results_folder = script_dir + "/results/" + save_label + "/" 
    os.makedirs(results_folder, exist_ok=True)
    optimizer = opt_func(model.parameters(), lr, **opt_kwargs)

    #scheduler = ExponentialLR(optimizer, gamma=0.9)

    # TODO: learning rate scheduling https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

    list_norm_of_batch_grad = []
    list_norm_of_aux_batch_grad = []
    list_norm_of_batch_grad_change = []
    list_running_avg_norm_of_batch_grad = []
    list_norm_of_batch_grad_running_avg = []
    list_training_loss = []
    list_val_loss = []
    list_val_accuracy = []
    list_training_time = []
    tensor_accum_grad = torch.Tensor().cpu() # TODO: shouldn't save this on the GPU
    tensor_model_params = torch.Tensor().cpu()

    list_alloc_mem, list_reserved_mem, list_free_reserved_mem = [], [], []

    prev_gradient = None
    prev_gradients = torch.Tensor().cpu() 
    
    quantities_to_record = ['norm_of_batch_grad', 'norm_of_aux_batch_grad', 'norm_of_batch_grad_change', \
    'running_avg_norm_of_batch_grad', 'norm_of_batch_grad_running_avg', 'training_loss', 'accum_grad', 
    'model_params', 'val_acc', 'val_loss', 'time_per_epoch', 'alloc_mem', 'reserved_mem', 'free_reserved_mem']

    # TODO: when I log stuff, should specify epoch / batch as indices

    for quantity in quantities_to_record:
        os.makedirs(results_folder + quantity + '/', exist_ok=True)

    for epoch in range(num_epochs):

        print('starting epoch {}'.format(epoch))

        tic_epoch_start = time.time()

        aux_indices = random.sample(range(len(train_ds)), aux_batch_size)
        auxiliary_ds = Subset(train_ds, aux_indices) 
        aux_loader = DataLoader(auxiliary_ds, len(auxiliary_ds))
        for aux_data in aux_loader:
            aux_input, aux_output = aux_data
        aux_input = aux_input.to(device)
        aux_output = aux_output.to(device)
        
        batch_counter = 0

        accum_grad = None

        
        # Training Phase 
        training_loss = 0.0
        for train_input, train_output in train_loader:
            batch_counter += 1
            batch_history_index = epoch*len(train_loader) + batch_counter

            train_input, train_output = train_input.to(device), train_output.to(device)

            loss = loss_fn(model(train_input), train_output) # generate predictions and calculate loss
            loss.backward() # compute gradients

            # compute auxiliary gradient for a larger batch size
            aux_model = copy.deepcopy(model)
            aux_loss = loss_fn(aux_model(aux_input), aux_output)
            aux_loss.backward()

            current_gradient = parameters_to_vector(j.grad for j in model.parameters()).cpu()
            current_aux_gradient = parameters_to_vector(j.grad for j in aux_model.parameters()).cpu()

            norm_current_gradient = torch.linalg.norm(current_gradient).item()
            norm_current_aux_gradient = torch.linalg.norm(current_aux_gradient).item()

            run['signals/batch/norm_batch_gradient'].log(norm_current_gradient)
            run['signals/batch/norm_auxiliary_batch_gradient'].log(norm_current_aux_gradient)
            run['signals/batch/squared_diff_batch_gradient'].log(norm_current_gradient**2 - norm_current_aux_gradient**2)

            list_norm_of_batch_grad.append(norm_current_gradient)
            list_norm_of_aux_batch_grad.append(norm_current_aux_gradient)

            # accumulate the gradient
            if accum_grad is None:
                accum_grad = copy.deepcopy(current_gradient)
            else:
                accum_grad = copy.deepcopy(accum_grad + current_gradient)

            optimizer.step() 
            optimizer.zero_grad() 

            training_loss += loss.item()
            #prof.step() # Pytorch profiler

            if prev_gradient is not None:
                
                run['signals/batch/cosine_similarity'].log(F.cosine_similarity(prev_gradient, current_gradient, dim=0))

                norm_grad_change = torch.linalg.norm(current_gradient - prev_gradient).item()
                
                run['signals/batch/norm_of_change_in_batch_gradients'].log(norm_grad_change)
                list_norm_of_batch_grad_change.append(norm_grad_change)

                signal_1 = norm_current_gradient**2 - 1/2 * norm_grad_change**2
                run['signals/batch/signal_1'].log(signal_1)
                # TODO: are there ways to validate (hopefully) that the latter term is small?
                signal_2 = signal_1 / (norm_current_gradient**2)
                run['signals/batch/signal_2'].log(signal_2)
            else:
                list_norm_of_batch_grad_change.append(np.nan)
                

            prev_gradient = copy.deepcopy(current_gradient)
            #print('device of prev_gradient', prev_gradient.device)

            # TODO: size of individual tensors:
            # "the size of a tensor a in memory is a.element_size() * a.nelement()."
            #print('prev_gradient size: ', prev_gradient.element_size() * prev_gradient.nelement())

            # keep a list of a few previous gradients
            prev_gradients = copy.deepcopy(torch.cat((prev_gradients, current_gradient.unsqueeze(0))))[-window_size:, :]
            norm_of_running_avg_of_grad = torch.linalg.norm(torch.mean(prev_gradients, dim = 0)).item()
            running_avg_of_norm_grad = torch.mean(torch.linalg.norm(prev_gradients, dim=1)).item()
            running_avg_of_squared_norm_grad = torch.mean(torch.linalg.norm(prev_gradients, dim=1)**2).item()
            #get_memory_info()
            #print('prev_gradients size: ', prev_gradients.element_size() * prev_gradients.nelement())
            #print('prev_gradients storage: ', sys.getsizeof(prev_gradients.storage()))
            
            run['metrics/batch/norm_of_running_avg_of_gradient'].log(norm_of_running_avg_of_grad)
            run['metrics/batch/running_avg_of_norm_of_gradient'].log(running_avg_of_norm_grad)

            list_norm_of_batch_grad_running_avg.append(norm_of_running_avg_of_grad)
            list_running_avg_norm_of_batch_grad.append(running_avg_of_norm_grad)

            signal_3 = window_size * norm_of_running_avg_of_grad**2 - running_avg_of_squared_norm_grad
            run['metrics/batch/signal3'].log(signal_3)

            prof.step()
            a, r, f = get_memory_info(log=True)
            list_alloc_mem.append(a)
            list_reserved_mem.append(r)
            list_free_reserved_mem.append(f)

            np.save(results_folder + 'alloc_mem/alloc_mem_' + str(trial) + '.npy', np.array(list_alloc_mem))
            np.save(results_folder + 'reserved_mem/reserved_mem_' + str(trial) + '.npy', np.array(list_reserved_mem))
            np.save(results_folder + 'free_reserved_mem/free_reserved_mem_' + str(trial) + '.npy', np.array(list_free_reserved_mem))

        #scheduler.step()

        toc_epoch_end = time.time()
        list_training_time.append(toc_epoch_end - tic_epoch_start)
        np.save(results_folder + 'time_per_epoch/time_per_epoch_' + str(trial) + '.npy', np.array(list_training_time))

        training_loss /= len(train_loader)
        run['metrics/epoch/training_loss'].log(training_loss)
        run['metrics/epoch/accum_gradient'].log(torch.linalg.norm(accum_grad))

        list_training_loss.append(training_loss)
        tensor_accum_grad = torch.cat([tensor_accum_grad, accum_grad], 0)
        tensor_model_params = torch.cat([tensor_model_params, parameters_to_vector(model.parameters()).cpu()], 0)   

        # save data as you go
        for record_name in ['norm_of_batch_grad', 'norm_of_aux_batch_grad', 'norm_of_batch_grad_change', \
            'running_avg_norm_of_batch_grad', 'norm_of_batch_grad_running_avg', 'training_loss']:
            os.makedirs(results_folder + record_name+'/', exist_ok=True)
            np.save(results_folder + record_name + '/' + record_name + '_' + str(trial) + '.npy', \
                np.array(eval('list_'+record_name)))

        np.save(results_folder + 'accum_grad/accum_grad_' + str(trial) + '.npy', tensor_accum_grad.detach().numpy())
        np.save(results_folder + 'model_params/model_params_' + str(trial) + '.npy', tensor_model_params.detach().numpy())

        # Validation phase
        validation_loss = 0.0
        validation_acc = 0.0
        for val_input, val_output in val_loader:

            val_input, val_output = val_input.to(device), val_output.to(device)
            with torch.no_grad():
                pred_output = model(val_input)

            val_loss = loss_fn(pred_output, val_output).item()
            validation_loss += val_loss

            val_acc = accuracy(pred_output, val_output)
            validation_acc += val_acc

        validation_loss /= len(val_loader)
        validation_acc /= len(val_loader)
        run['metrics/epoch/val_loss'].log(validation_loss)
        run['metrics/epoch/val_acc'].log(validation_acc)

        list_val_accuracy.append(validation_acc)
        list_val_loss.append(validation_loss)
    
        np.save(results_folder + 'val_acc/val_acc_' + str(trial) + '.npy', np.array(list_val_accuracy))
        np.save(results_folder + 'val_loss/val_loss_' + str(trial) + '.npy', np.array(list_val_loss))

        # save model
        model_path = results_folder + 'model_epoch_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_path)
        # run['model_checkpoints/my_model'].upload('model_checkpoints/my_model.pt')



        print('finished training and validating epoch {}, took {} seconds'.format(epoch, time.time()-tic_epoch_start))


def early_stopping(metric, window, tolerance, decrease = True):
    # TODO: implement early stopping according to some criterion, e.g., val loss does not decrease

    # if some funcion (metric[-window:], decrease) < tolerance:
    # return True, i.e., the metric is not improving

    pass


def cross_validation(data, folds):
    pass


