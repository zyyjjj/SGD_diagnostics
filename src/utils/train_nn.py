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
        writer,
        prof, 
        opt_kwargs={},
        window_size = 10):

    start_time = time.time()

    script_dir = os.path.abspath(os.getcwd())
    results_folder = script_dir + "/results/" + save_label + "/" 

    os.makedirs(results_folder, exist_ok=True)

    optimizer = opt_func(model.parameters(), lr, **opt_kwargs)
    # TODO: where is it initialized at?
    # TODO: could it be initialized at a saddle point that is hard to escape at small learning rates?
    # TODO: just compute the loss and gradient values at some randomly chosen points
    # TODO: think: what does the surface of cross-entropy loss look like?

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
    tensor_accum_grad = torch.Tensor().to(device) # TODO: shouldn't save this on the GPU
    tensor_model_params = torch.Tensor().to(device)


    quantities_to_record = ['norm_of_batch_grad', 'norm_of_aux_batch_grad', 'norm_of_batch_grad_change', \
    'running_avg_norm_of_batch_grad', 'norm_of_batch_grad_running_avg', 'training_loss', 'accum_grad', 
    'model_params', 'val_acc', 'val_loss', 'time_per_epoch']

    # TODO: when I log stuff, should specify epoch / batch as indices

    for quantity in quantities_to_record:
        os.makedirs(results_folder + quantity + '/', exist_ok=True)

    for epoch in range(num_epochs):

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
        prev_gradient = None
        prev_gradients = torch.Tensor().to(device) # TODO: don't save this on the GPU
        
        # Training Phase 
        training_loss = 0.0
        for train_input, train_output in train_loader:
            batch_counter += 1
            batch_history_index = epoch*len(train_loader) + batch_counter

            train_input, train_output = train_input.to(device), train_output.to(device)
            #train_input = train_input.to(device)
            #train_output = train_output.to(device)

            loss = loss_fn(model(train_input), train_output) # generate predictions and calculate loss
            loss.backward() # compute gradients

            # compute auxiliary gradient for a larger batch size
            aux_model = copy.deepcopy(model)
            aux_loss = loss_fn(aux_model(aux_input), aux_output)
            aux_loss.backward()

            current_gradient = parameters_to_vector(j.grad for j in model.parameters())
            current_aux_gradient = parameters_to_vector(j.grad for j in aux_model.parameters())
            # TODO: hypothesis: these carry histories with them, but I only need the values
            print('device of current_gradient', current_gradient.device)


            norm_current_gradient = torch.linalg.norm(current_gradient).item()
            #print('norm_current_gradient', norm_current_gradient, norm_current_gradient.device, norm_current_gradient.item())
            norm_current_aux_gradient = torch.linalg.norm(current_aux_gradient).item()

            writer.add_scalar('norm of current batch gradient', norm_current_gradient, batch_history_index)
            writer.add_scalar('norm of auxiliary gradient', norm_current_aux_gradient, batch_history_index)
            writer.add_scalar('difference in squared norm of current batch gradient and auxiliary gradient', \
                norm_current_gradient**2 - norm_current_aux_gradient**2, batch_history_index)

            list_norm_of_batch_grad.append(norm_current_gradient)
            list_norm_of_aux_batch_grad.append(norm_current_aux_gradient)

            # accumulate the gradient
            if accum_grad is None:
                accum_grad = copy.deepcopy(current_gradient)
            else:
                accum_grad = copy.deepcopy(accum_grad + current_gradient)
                # TODO: This step could be using too much memory
            print('device of accumulated gradient', accum_grad.device)
            get_memory_info()

            optimizer.step() 
            optimizer.zero_grad() 

            training_loss += loss.item()
            #prof.step() # Pytorch profiler

            if prev_gradient is not None:
                writer.add_scalar('cosine similarity between previous and current batch gradients', \
                    F.cosine_similarity(prev_gradient, current_gradient, dim=0), batch_history_index)
                norm_grad_change = torch.linalg.norm(current_gradient - prev_gradient).item()
                writer.add_scalar('norm of change in batch gradients', norm_grad_change, batch_history_index)
                list_norm_of_batch_grad_change.append(norm_grad_change)

                signal_1 = norm_current_gradient**2 - 1/2 * norm_grad_change**2
                writer.add_scalar('signal 1 = ||batch gradient||^2 - 1/2 ||change in batch gradient||^2', signal_1, batch_history_index)
                # TODO: are there ways to validate (hopefully) that the latter term is small?
                signal_2 = signal_1 / (norm_current_gradient**2)
                writer.add_scalar('signal 2 = 1 - 1/2 ||change in batch gradient||^2 / ||batch gradient||^2', signal_2, batch_history_index)

            prev_gradient = copy.deepcopy(current_gradient)
            print('device of prev_gradient', prev_gradient.device)
            get_memory_info()

            # TODO: size of individual tensors:
            # "the size of a tensor a in memory is a.element_size() * a.nelement()."
            print('prev_gradient size: ', prev_gradient.element_size() * prev_gradient.nelement())

            # keep a list of a few previous gradients
            # TODO: current_gradient is likely on the GPU, we only need its value; prev_gradients.cpu()? 
            prev_gradients = copy.deepcopy(torch.cat((prev_gradients, current_gradient.unsqueeze(0))))[-window_size:, :].to(device)
            #prev_gradients = prev_gradients[-window_size:, :]
            norm_of_running_avg_of_grad = torch.linalg.norm(torch.mean(prev_gradients, dim = 0)).item()
            running_avg_of_norm_grad = torch.mean(torch.linalg.norm(prev_gradients, dim=1)).item()
            running_avg_of_squared_norm_grad = torch.mean(torch.linalg.norm(prev_gradients, dim=1)**2).item()
            print('two norm signals, checking correctness: ', running_avg_of_norm_grad, running_avg_of_squared_norm_grad)
            #print('device of norm signals', norm_of_running_avg_of_grad.device, running_avg_of_norm_grad.device)
            get_memory_info()
            print('prev_gradients size: ', prev_gradients.element_size() * prev_gradients.nelement())
            print('prev_gradients storage: ', sys.getsizeof(prev_gradients.storage()))
            
            writer.add_scalar('norm of running avg of gradient', norm_of_running_avg_of_grad, batch_history_index)
            writer.add_scalar('running avg of norm of gradient', running_avg_of_norm_grad, batch_history_index)

            list_norm_of_batch_grad_running_avg.append(norm_of_running_avg_of_grad)
            list_running_avg_norm_of_batch_grad.append(running_avg_of_norm_grad)

            signal_3 = window_size * norm_of_running_avg_of_grad**2 - running_avg_of_squared_norm_grad
            writer.add_scalar('signal 3 = window_size * sq norm of running avg of gradient - running avg of sq norm of grad', signal_3, batch_history_index)


            prof.step()
        #scheduler.step()

        # TODO: check: when we time an epoch in training, do we include the validation part? or just the training part
        toc_epoch_end = time.time()
        list_training_time.append(toc_epoch_end - tic_epoch_start)
        np.save(results_folder + 'time_per_epoch/time_per_epoch_' + str(trial) + '.npy', np.array(list_training_time))

        training_loss /= len(train_loader)
        writer.add_scalar('training loss', training_loss, epoch)
        writer.add_scalar('accumulated gradient over batches', torch.linalg.norm(accum_grad), epoch)

        list_training_loss.append(training_loss)
        tensor_accum_grad = torch.cat([tensor_accum_grad, accum_grad], 0)
        # TODO: double check whether this is the right way to save weights
        tensor_model_params = torch.cat([tensor_model_params, parameters_to_vector(model.parameters())], 0) 
        print('model params tensor size: ', tensor_model_params.element_size() * tensor_model_params.nelement())
        print('model params tensor storage: ', sys.getsizeof(tensor_model_params.storage()))
            

        # save data as you go
        for record_name in ['norm_of_batch_grad', 'norm_of_aux_batch_grad', 'norm_of_batch_grad_change', \
            'running_avg_norm_of_batch_grad', 'norm_of_batch_grad_running_avg', 'training_loss']:
            os.makedirs(results_folder + record_name+'/', exist_ok=True)
            #print(type(eval('list_'+record_name)))
            #print(eval('list_'+record_name).device)
            np.save(results_folder + record_name + '/' + record_name + '_' + str(trial) + '.npy', \
                np.array(eval('list_'+record_name)))

        np.save(results_folder + 'accum_grad/accum_grad_' + str(trial) + '.npy', tensor_accum_grad.detach().cpu().numpy())
        np.save(results_folder + 'model_params/model_params_' + str(trial) + '.npy', tensor_model_params.detach().cpu().numpy())

        # Validation phase
        validation_loss = 0.0
        validation_acc = 0.0
        for val_input, val_output in val_loader:
            val_input, val_output = val_input.to(device), val_output.to(device)
            #val_input = val_input.to(device)
            #val_output = val_output.to(device)

            with torch.no_grad():
                pred_output = model(val_input)

            val_loss = loss_fn(pred_output, val_output).item()
            validation_loss += val_loss

            val_acc = accuracy(pred_output, val_output)
            validation_acc += val_acc

        validation_loss /= len(val_loader)
        validation_acc /= len(val_loader)
        writer.add_scalar('avg validation loss', validation_loss, epoch)
        writer.add_scalar('avg validation accuracy', validation_acc, epoch)

        list_val_accuracy.append(validation_acc)
        list_val_loss.append(validation_loss)
    
        np.save(results_folder + 'val_acc/val_acc_' + str(trial) + '.npy', np.array(list_val_accuracy))
        np.save(results_folder + 'val_loss/val_loss_' + str(trial) + '.npy', np.array(list_val_loss))


    print('finished training {} epochs, took {} seconds'.format(num_epochs, time.time()-start_time))


def early_stopping(criterion, tolerance):
    # TODO: implement early stopping according to some criterion, e.g., val loss does not decrease
    pass