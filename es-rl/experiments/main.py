import argparse
import datetime
import os
import pickle
import platform

import random
import gym
import IPython
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from context import es, utils
from es.algorithms import GA, ES, NES, sES, sNES, xNES
from es.envs import create_gym_environment
from es.eval_funs import gym_render, gym_rollout, gym_test, supervised_eval, supervised_test
from es.models import *
from torchvision import datasets, transforms
from utils.misc import get_inputs_from_dict, get_inputs_from_dict_class


def parse_inputs():
    """
    This method parses inputs given through the terminal using `argparse`.
    
    It assigns values to all parameters of the algorithms and their models,
    environments, optimizers and learning rate schedulers. It also defines
    behaviour related to the execution of the code through the flags
    `test`, `restore`, `cuda` and `silent`.
    
    If a value is not specified, it defaults to the default value also defined here.

    Learning rate decay: 
        A learning rate decay, gamma, will result in a fraction rho of the original
        learning rate after 10000 generations.
        gamma       rho
        0.99954     1%
        0.99970     5%
        0.99977     10%

    Algorithm hyperparameter settings
        --algorithm ES --optimizer SGD --lr 0.020 --cov_lr 0.2 --gamma 0.99970 --perturbations 40 --sigma 0.05
        --algorithm ES --optimizer SGD --lr 0.050 --cov_lr 0.5 --gamma 0.99970 --perturbations 100 --sigma 0.05
        --algorithm ES --optimizer SGD --lr 0.100 --cov_lr 1.0 --gamma 0.99970 --perturbations 100 --sigma 0.05
        --algorithm ES --optimizer SGD --lr 0.200 --cov_lr 2.0 --gamma 0.99970 --perturbations 100 --sigma 0.05
        --algorithm ES --optimizer SGD --lr 0.080 --gamma 0.99970 --perturbations 300 --sigma 0.05
        --algorithm ES --optimizer SGD --lr 0.200 --gamma 0.99970 --perturbations 1000 --sigma 0.05
    """
    sigma_choices = ['None', 'single', 'per-layer', 'per-weight']
    sm_choices = ['None', 'ABS', 'SUM', 'SO', 'R']

    parser = argparse.ArgumentParser(description='Experiments')
    # Algorithm
    parser.add_argument('--algorithm', type=str, default='NES', metavar='ALG', help='Model name in es.models')
    parser.add_argument('--perturbations', type=int, default=100, metavar='N', help='Number of perturbed models to make; must be even')
    parser.add_argument('--sigma', type=float, default=0.05, metavar='SD', help='Initial noise standard deviation')
    parser.add_argument('--optimize-sigma', type=str, default='True', choices=sigma_choices, metavar='OS', help='Which type of covariance matrix parameterization to use')
    parser.add_argument('--no-antithetic', action='store_true', default=True, help='Boolean to not to use antithetic sampling')
    parser.add_argument('--forced-refresh', type=float, default=0.0, help='Forced refresh rate for importance sampling')
    parser.add_argument('--adaptation-sampling', action='store_true', default=False, help='Whether or not to use adaptive sampling for updating learning rate')
    parser.add_argument('--common-random-numbers', action='store_true', default=False, help='Whether or not to use common random numbers for perturbation evaluation')
    parser.add_argument('--safe-mutation', type=str, default='None', choices=sm_choices, help='String denoting the type of safe mutations to use')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='BS', help='Batch size agent evaluation (max episode steps for RL setting rollouts)')
    parser.add_argument('--max-generations', type=int, default=5, metavar='MG', help='Maximum number of generations')
    parser.add_argument('--val-every', type=int, default=25, metavar='TE', help='Interval at which to test the model on validation set (if relevant)')
    # Environment
    parser.add_argument('--env-name', type=str, default='MNIST', metavar='ENV', help='RL environment or dataset')
    parser.add_argument('--frame-size', type=int, default=84, metavar='FS', help='Square size of frames in pixels')
    # Model
    parser.add_argument('--model', type=str, default='MNISTNet', metavar='MOD', help='Model name in es.models')
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR', help='Optimizer learning rate')
    parser.add_argument('--cov-lr', type=float, default=None, metavar='LR_BETA', help='Optimizer learning rate for covariance parameter')
    parser.add_argument('--momentum', type=float, default=0.9, help='Optimizer momentum')
    parser.add_argument('--nesterov', action='store_true', help='Boolean to denote if optimizer momentum is Nesterov')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='Optimizer L2 norm weight decay penalty')
    # Learning rate scheduler
    parser.add_argument('--lr-scheduler', type=str, default='ExponentialLR', help='Learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=1, help='Learning rate decay rate')
    parser.add_argument('--factor', type=float, default=0.8, help='Reduction factor [ReduceLROnPlateau]')
    parser.add_argument('--patience', type=int, default=100, help='Patience before lowering learning rate [ReduceLROnPlateau]')
    parser.add_argument('--threshold', type=float, default=1e-4, help='Threshold for comparing best to current [ReduceLROnPlateau]')
    parser.add_argument('--cooldown', type=int, default=50, help='Cooldown after lowering learning rate before able to do it again [ReduceLROnPlateau]')
    parser.add_argument('--mode', type=str, default='max', help='Optimization mode (minimization or maximization) [ReduceLROnPlateau]')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimal learning rate [ReduceLROnPlateau]')
    parser.add_argument('--milestones', type=list, default=50, help='Milestones on which to lower learning rate[MultiStepLR]')
    parser.add_argument('--step-size', type=int, default=50, help='Step interval on which to lower learning rate[StepLR]')
    # Execution
    parser.add_argument('--workers', type=int, default=mp.cpu_count(), help='Interval in seconds for saving checkpoints')
    parser.add_argument('--chkpt-int', type=int, default=600, help='Interval in seconds for saving checkpoints')
    parser.add_argument('--track-parallel', action='store_true', help='Whether to track evaluation of perturbations in each iteration')
    parser.add_argument('--test', action='store_true', help='Test the model (accuracy or env render), no training')
    parser.add_argument('--id', type=str, default='test', metavar='ID', help='ID of the this run. Appended as folder to path as checkpoints/<ID>/ if not empty')
    parser.add_argument('--restore', type=str, default='', metavar='RES', help='Checkpoint from which to restore')
    parser.add_argument('--use-new-algorithm', action='store_true', help='Whether to use a new algorithm setting on the restored checkpoint')
    parser.add_argument('--cuda', action='store_true', default=False, help='Enables CUDA training')
    parser.add_argument('--silent', action='store_true', help='Silence print statements during training')
    parser.add_argument('--do-permute-train-labels', action='store_true', help='Permute the training labels randomly')
    parser.add_argument('--lr-from-perturbations', type=int, default=0, help='Get the learning rate heuristically from the number of perturbations')
    
    ###LORENZO ADDITIONS###
    #Baseline for variance reduction
    parser.add_argument('--baseline_mu', action='store_true', default=True, help='Use baseline for the mu parameter')
    parser.add_argument('--baseline_sigma', action='store_true', default=True, help='Use baseline for the sigma parameter')
    parser.add_argument('--small_sigma', action='store_true', default=False, help='If using baseline, use small sigma approximation')

    #Use natural gradient
    parser.add_argument('--use-naturgrad', action='store_true', default=False, help='Use natural gradient')

    args = parser.parse_args()
    return args


def validate_inputs(args):
    """
    This method validates the given inputs from `argparse.ArgumentParser.parse_args()`.
    """
    # Input validation
    assert args.no_antithetic or args.perturbations % 2 == 0             # Even number of perturbations if using antithetic sampling
    assert not args.test or (args.test and args.restore)                # Testing requires restoring a model
    assert not args.cuda or (args.cuda and torch.cuda.is_available())   # Can only use CUDA if avaiable

    # Convert None strings to NoneType
    for a, v in args.__dict__.items():
        if v == 'None':
            args.__dict__[a] = None
    
    # Determine supervised/reinforcement learning problem
    if args.env_name in gym.envs.registry.env_specs.keys():
        args.is_supervised = False
        args.is_rl = True
    else:
        args.is_supervised = True
        args.is_rl = False


def create_model(args):
    # Create model
    ModelClass = getattr(es.models, args.model)
    # Supervised or RL
    if args.is_rl:
        args.model = ModelClass(args.env.observation_space, args.env.action_space)
    elif args.is_supervised:
        args.model = ModelClass()
        if args.env_name == 'MNIST':
            assert ModelClass in [MNISTNet, MNISTNetNoBN, MNISTNetNoInit, MNISTNetDropout]
        elif args.env_name == 'FashionMNIST':
            assert ModelClass in [MNISTNet, MNISTNetNoBN, MNISTNetNoInit, MNISTNetDropout]
        elif args.env_name == 'CIFAR10':
            assert ModelClass in [CIFARNet]
    assert type(args.model) is not str
    # CUDA
    if args.cuda:
        args.model = args.model.cuda()


def create_optimizer(args):
    # Parameters to optimize are model parameters that require gradient and sigma if chosen
    opt_pars = []
    opt_pars.append({'label': 'model_params', 'params': args.model.parameters()})
    # Create optimizer
    OptimizerClass = getattr(optim, args.optimizer)
    optimizer_input_dict = get_inputs_from_dict(OptimizerClass.__init__, vars(args))
    args.optimizer = OptimizerClass(opt_pars, **optimizer_input_dict)


def create_lr_scheduler(args):
    if args.lr_scheduler is not None:
        # Create learning rate scheduler
        SchedulerClass = getattr(optim.lr_scheduler, args.lr_scheduler)
        scheduler_input_dict = get_inputs_from_dict(SchedulerClass.__init__, vars(args))
        # Set mode to maximization if the scheduler is `ReduceLRONPlateau`
        if SchedulerClass is optim.lr_scheduler.ReduceLROnPlateau:
            scheduler_input_dict['mode'] = 'max'
        args.lr_scheduler = SchedulerClass(**scheduler_input_dict)
    else:
        args.lr_scheduler = None


def create_environment(args):
    if args.is_rl:
        args.env = create_gym_environment(args.env_name, sqaure_size=args.frame_size)
    elif args.is_supervised:
        # Create normalization transforms (values computed by torchutils.dataset_mean_and_var)
        if args.env_name == 'MNIST':
            data_set = datasets.MNIST
            mean = (0.130660742521286,)
            var = (0.30810874700546265,)
        elif args.env_name == 'FashionMNIST':
            data_set = datasets.FashionMNIST
            mean = (-2.9279166483320296e-05,)
            var = (34.189090728759766,)
        elif args.env_name == 'CIFAR10':
            data_set = datasets.CIFAR10
            mean = (0.48753172159194946, 0.47322431206703186, 0.4359692335128784)
            var = (0.24663160741329193, 0.24822315573692322, 0.2677987813949585)
        elif args.env_name == 'CIFAR100':
            data_set = datasets.CIFAR100
            mean = (0.5141649842262268, 0.47902533411979675, 0.4298681914806366)
            var = (0.2685449421405792, 0.26044416427612305, 0.28062567114830017)
        # Arguments for data set
        data_dir = os.path.join(args.file_path, 'data', args.env_name)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, var)])
        train_set_kwargs = {'train': True, 'download': True, 'transform': transform}
        test_set_kwargs = {'train': False, 'download': True, 'transform': transform}
        # Create dataset and DataLoader
        train_set = data_set(data_dir, **train_set_kwargs)
        test_set = data_set(data_dir, **test_set_kwargs)
        # Arguments for data loader
        batch_size = args.batch_size if not(args.test) else 1000
        train_loader_kwargs = {'num_workers': 0, 'batch_size': batch_size, 'shuffle': True}
        test_loader_kwargs = {'num_workers': 0, 'batch_size': len(test_set), 'shuffle': True}
        if args.cuda:
            train_loader_kwargs['pin_memory'] = True
        if not args.test and args.do_permute_train_labels:
            # Permute the labels randomly to test 'Rethinking Generalization'
            train_set.train_labels = torch.LongTensor(np.random.permutation(train_set.train_labels))
        args.env = torch.utils.data.DataLoader(train_set, **train_loader_kwargs)
        args.val_env = torch.utils.data.DataLoader(test_set, **test_loader_kwargs)
    
    assert hasattr(args, 'env')


def create_algorithm(args):
    AlgorithmClass = getattr(es.algorithms, args.algorithm)
    algorithm_input_dict = get_inputs_from_dict_class(AlgorithmClass, vars(args), recursive=True)
    args.algorithm = AlgorithmClass(**algorithm_input_dict)


def create_checkpoint(args):
    # Create checkpoint directory if not restoring
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S.%f")
    info_str = str(args.algorithm) + '|' + args.env_name + '|' + args.model.__class__.__name__
    args.chkpt_dir = os.path.join(args.file_path, 'checkpoints', args.id, '{:s}|{:s}'.format(timestamp, info_str))
    if not os.path.exists(args.chkpt_dir):
        os.makedirs(args.chkpt_dir)
    # AlgorithmClass = getattr(es.algorithms, args.algorithm)
    # algorithm_input_dict = get_inputs_from_dict_class(AlgorithmClass, vars(args), recursive=True)
    # exclude_keys = ['chkpt_int','chkpt_dir','env','eval_fun','lr_scheduler','model','optimizer','silent']
    # include_keys = sorted(list(set(algorithm_input_dict.keys()).difference(exclude_keys)))
    # for k in include_keys:
    #     info_str += '|' + str(k) + '-' + str(algorithm_input_dict[k])
 

def get_eval_funs(args):
    if args.is_rl:
        args.eval_fun = gym_rollout
        args.test_fun = gym_test
        args.rend_fun = gym_render
    elif args.is_supervised:
        args.eval_fun = supervised_eval
        args.test_fun = supervised_test


def get_lr_from_perturbations(args):
    # Problem dimension from perturbations
    d = np.exp((args.perturbations-4)/3)
    if args.lr_from_perturbations == 1:
        args.lr = (9 + 3 * np.log(d))/(5*d**(3/2))
    elif args.lr_from_perturbations == 2:
        #args.lr = (3 + np.log(d))/(5*d**(1/2))
        args.lr = (3 + np.log(d))/(30*d**(1/2))
    elif args.lr_from_perturbations == 3:
        args.lr = d**(1/2)/5*(3 + np.log(d))
    return args

def test_model(args):
    args.algorithm.model.eval()
    if args.cuda:
        args.algorithm.model.cuda()
    if args.is_rl:
        # args.test_fun(args.algorithm.model, args.env, max_episode_length=args.batch_size, n_episodes=100, chkpt_dir=args.chkpt_dir)
        args.rend_fun(args.algorithm.model, args.env, max_episode_length=args.batch_size)
    else:
        args.test_fun(args.algorithm.model, args.val_env, cuda=args.cuda, chkpt_dir=args.chkpt_dir)
        #args.rend_fun(args.algorithm.mode, args.env, max_episode_length=args.batch_size)


if __name__ == '__main__':
    #TODO: manual seed, te moreve when sending to queue
    master_seed = 42
    random.seed(master_seed)
    torch.manual_seed(master_seed)
    np.random.seed(master_seed)
    
    # Parse and validate
    args = parse_inputs()
    validate_inputs(args)
    args.file_path = os.path.split(os.path.realpath(__file__))[0]
    # Get hyper parameters
    if args.lr_from_perturbations:
        args = get_lr_from_perturbations(args)
    # Create environment, model, optimizer and learning rate scheduler
    create_environment(args)
    create_model(args)
    create_optimizer(args)
    create_lr_scheduler(args)
    # Get functions for evaluation and testing
    get_eval_funs(args)
    # Create checkpoint
    if args.restore:
        args.chkpt_dir = args.restore
    else:
        create_checkpoint(args)
    # Create algorithm
    create_algorithm(args)
    if args.restore:
        args.algorithm.load_checkpoint(args.restore, load_best=False, load_algorithm=(not args.use_new_algorithm))
    
    # Set number of OMP threads for CPU computations
    # NOTE: This is needed for my personal stationary Linux PC for partially unknown reasons
    # if platform.system() == 'Linux':
    #     torch.set_num_threads(1)

    # Execute
    if not args.test:
        try:
            args.algorithm.train()
        except KeyboardInterrupt:
            print("Training stopped by user.")
        finally:
            args.test = True
            create_environment(args)

    # Try to load best, then latest, checkpoint and run test
    try:
        args.algorithm.load_checkpoint(args.chkpt_dir, load_best=True)
    except FileNotFoundError as best_e:
        print("Could not load best model after training: ", best_e)
        try:
            args.algorithm.load_checkpoint(args.chkpt_dir, load_best=False)
        except FileNotFoundError as e:
            print("Could not load latest model after training: ", e)
        else:
            test_model(args)
    else:
        test_model(args)


"""
Fashion-MNIST
-----------------
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
"""
