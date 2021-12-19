import argparse, itertools

def parse_hp_args():
    parser = argparse.ArgumentParser(description='input names and values of hyperparameters to vary')
    parser.add_argument('-o', '--optimizer', default = 'SGD', help='optimizer algorithm')
    parser.add_argument('-op', '--optimizer_params', action='append', 
            help='optimizer parameters')
    parser.add_argument('-ov', '--optimizer_param_values', nargs='+', action='append', type=float,
            help='values of optimizer parameters')
    parser.add_argument('-lr', '--learning_rates', nargs='+', action='append', type = float, help='learning rates to try')
    parser.add_argument('-b', '--batch_sizes', nargs='+', action='append', type = int, help = 'batch sizes to try', required=True)
    parser.add_argument('--aux_batch_size', type = int, default = 1024)
    parser.add_argument('-T', '--num_epochs', default=500, type=int, help='number of training epochs')
    #parser.add_argument('-s', '--scheduling', action='store_true', help='run linear scheduling of learning rate')
    parser.add_argument('--trial', type=int, help = 'specify trial number, which also serves as random seed')

    args = parser.parse_args()
    
    configs_set = list(itertools.product(args.learning_rates[0], args.batch_sizes[0]))
    print('configs_set {}'.format(configs_set))

    if args.optimizer_params is not None:
        opt_kwargs = dict(zip(args.optimizer_params, args.optimizer_param_values[0]))
    else:
        opt_kwargs = dict()

    return args, opt_kwargs, configs_set

