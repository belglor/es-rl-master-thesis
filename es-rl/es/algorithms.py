import collections
import copy
import csv
import datetime
import math
import os
import pickle
import platform
import pprint
import queue
import time
from abc import ABCMeta, abstractmethod
from functools import partial
from itertools import zip_longest

import gym
import IPython
import numpy as np
import pandas as pd
import torch
import torch.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable

from context import utils
from utils.misc import get_inputs_from_dict, to_numeric, isint
from utils.plotting import plot_stats
from utils.progress import PoolProgress
from utils.torchutils import summarize_model

#%%
class Algorithm(object): 
    """Abstract class for variational algorithms

    Attributes:
        model (torch.nn): A pytorch module model
        env (gym): A gym environment
        optimizer (torch.optim.Optimizer): A pytorch optimizer
        lr_scheduler (torch.optim.lr_scheduler): A pytorch learning rate scheduler
        perturbations (int): The number of perturbed models to evaluate
        batch_size (int): The number of observations for an evaluate (supervised)
        max_generations (int): The maximum number of generations to train for
        safe_mutation (str): The version of safe mutations to use. Valid options are `ABS`, `SUM` and `SO`
        no_antithetic (bool): If `True`, the algorithm samples without also taking the antithetic sample
        chktp_dir (str): The directory to use to save/load checkpoints. If not absolute, it will be appended without overlap to the path of this file when executing
        chkpt_int (int): The interval in seconds between checkpoint saves. If chkpt_int<=0, a checkpoint is made at every generation.
        cuda (bool): Boolean to denote whether or not to use CUDA
        silent (bool): Boolean to denote if executing should be silent (no terminal printing)
    """

    __metaclass__ = ABCMeta

    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, common_random_numbers=False, adaptation_sampling=True, forced_refresh=0.01, val_env=None, val_every=25, workers=mp.cpu_count(), chkpt_dir=None, chkpt_int=600, track_parallel=False, cuda=False, silent=False):
        self.algorithm = self.__class__.__name__
        # Algorithmic attributes
        self.model = model
        self.env = env
        self.val_env = val_env
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_fun = eval_fun
        self.safe_mutation = safe_mutation
        self.no_antithetic = no_antithetic
        self.adaptation_sampling = adaptation_sampling
        self.common_random_numbers = common_random_numbers
        self.perturbations = perturbations
        self.forced_refresh = forced_refresh  # Forced resampling rate for importance mixing
        self.batch_size = batch_size
        self.max_generations = max_generations
        self.val_every = val_every
        # Execution attributes
        self.workers = workers if workers <= mp.cpu_count() else mp.cpu_count()
#        self.workers = 1
        self.track_parallel = track_parallel
        self.cuda = cuda
        self.silent = silent
        # Checkpoint attributes
        self.chkpt_dir = chkpt_dir
        self.chkpt_int = chkpt_int
        # Safe mutation sensitivities
        n_layers = 0
        for _ in self.model.parameters():
            n_layers += 1
        self.sensitivities = [None]*n_layers
        # Get inputs for sensitivity calculations by sampling from environment or
        # getting a batch from the data loader
        if hasattr(self.env, 'observation_space'):
            # Preallocate for collecting..
            s = self.env.reset()
            self.sens_inputs = torch.zeros((1000, *s.shape))
            # Sample from environment
            # self.sens_inputs[0,...] = torch.from_numpy(s)
            # for i in range(999):
            #     self.sens_inputs[i+1,...] = torch.from_numpy(self.env.observation_space.sample())
        elif type(self.env) is torch.utils.data.DataLoader:
            self.sens_inputs = next(iter(self.env))[0]
        if self.safe_mutation is None:
            self.sens_inputs = self.sens_inputs[0:2] # self.sens_inputs[0].view(1, *self.sens_inputs.size()[1:])
        # Attributes to exclude from the state dictionary
        self.exclude_from_state_dict = {'env', 'optimizer', 'lr_scheduler', 'model', 'stats', 'sens_inputs'}
        # Initialize dict for saving statistics
        self._base_stat_keys = {'generations', 'walltimes', 'workertimes', 'unp_rank', 'n_reused', 'n_rejected', 'grad_norm', 'param_norm'}
        self.stats = {key: [] for key in self._base_stat_keys}
        self._training_start_time = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def perturb_model(self, seed):
        pass

    @abstractmethod
    def compute_gradients(self, *kwargs):
        pass

    def _add_parameter_to_optimize(self, parameter_group, overwrite=False):
        """Adds a parameter group to be optimized.

        The parameter group can consists of an arbitrary number of parameters that 
        will all get the same learning rate. The group is added to the optimizer and
        the lr_scheduler is reinitialized with the same inputs on the updated optimizer.

        If the given parameter_group has a label key, then this function returns a reference
        to the parameter group in the optimizer.
        
        Args:
            parameter_group (dict): The parameter group
        """
        # If a parameter group is labelled, check if already exists in order to delete
        if overwrite and 'label' in parameter_group:
            if len(list(filter(lambda group: group['label'] == parameter_group['label'], self.optimizer.param_groups))) > 0:
                self.optimizer.param_groups = [g for g in self.optimizer.param_groups if 'label' not in g or g['label'] != parameter_group['label']]
        # Add parameter group to optimizer and reinitialize lr scheduler
        self.optimizer.add_param_group(parameter_group)            
        inputs = get_inputs_from_dict(self.lr_scheduler.__class__.__init__, vars(self.lr_scheduler))
        self.lr_scheduler = self.lr_scheduler.__class__(**inputs)
        # Return reference if parameter group is labelled
        if 'label' in parameter_group:
            return list(filter(lambda group: group['label'] == parameter_group['label'], self.optimizer.param_groups))[0]

    @staticmethod
    def unperturbed_rank(returns, unperturbed_return):
        """Computes the rank of the unperturbed model among the perturbations.
        
        Args:
            returns (np.array): Returns of evaluated perturbed models.
            unperturbed_return (float): Return of the unperturbed model.
        
        Returns:
            int: Rank of the unperturbed model among the perturbations
        """      
        return (returns > unperturbed_return).sum() + 1

    @staticmethod
    def fitness_shaping(returns):
        """Computes the fitness shaped returns.

        Performs the fitness rank transformation used for CMA-ES.
        Reference: Natural Evolution Strategies [2014]
        
        Args:
            returns (np.array): Returns of evaluated perturbed models.
        
        Returns:
            np.array: Shaped returns
        """
        assert type(returns) == np.ndarray
        n = len(returns)
        sorted_indices = np.argsort(-returns)
        u = np.zeros(n)
        for k in range(n):
            u[sorted_indices[k]] = np.max([0, np.log(n / 2 + 1) - np.log(k + 1)])
        return u / np.sum(u) - 1 / n

    @staticmethod
    def fitness_normalization(returns, unperturbed_return):
        returns -= unperturbed_return
        if not np.isnan(returns.std()):
            returns /= returns.std()
        else:
            assert (returns == returns).all()
        IPython.embed()
        returns /= returns.sum()
        assert returns.sum() == 1.0
        return returns
    
    @staticmethod
    def get_perturbation(size, sensitivities=None, cuda=False):
        """Draws a perturbation tensor of dimension `size` from a standard normal.

        If `sensitivities is not None`, then the perturbation is scaled by these.
        If sensitivities are given and `sensitivities.size() == size[1:]` then the `size[0]` is 
        intepreted as a number of samples each of which is to be scaled by the given sensitivities.
        """
        if type(size) in [tuple, list]:
            size = torch.Size(size)
        elif type(size) is int:
            size = torch.Size([size])
        elif type(size) is torch.Size:
            pass
        else:
            raise TypeError("Input `size` must be of type `int`, `list`, `tuple` or `torch.Size` but was `{}`".format(type(size).__name__))
        if sensitivities is not None and sensitivities.size() == size[1:]:
            samples = size[0]
        assert sensitivities is None or sensitivities.size() == size or sensitivities.size() == size[1:], "Sensitivities must match size of perturbation"
        if cuda:
            eps = torch.cuda.FloatTensor(size)
        else:
            eps = torch.FloatTensor(size)
        eps.normal_(mean=0, std=1)
        if sensitivities is not None:
            if sensitivities.size() == size[1:]:
                for s in range(samples):
                    eps[s, ...] /= sensitivities      # Scale by sensitivities
                    if eps.numel() > 1:
                        eps /= eps.std()              # Rescale to unit variance
            else:
                eps /= sensitivities     # Scale by sensitivities
                if eps.numel() > 1:
                    eps /= eps.std()     # Rescale to unit variance
        return eps

    @staticmethod
    def scale_by_sensitivities(eps, sensitivities=None):
        if sensitivities is None:
            return eps
        if eps.numel() > 1:
            # Scale to unit variance
            std = eps.std()
            eps /= std
        # Scale by sensitivities
        eps /= sensitivities
        if eps.numel() > 1:
            # Rescale to unit variance
            eps /= eps.std()
            # Rescale to original variance
            eps *= std
        return eps

    def _vec2modelrepr(self, vec, only_trainable=True, yield_generator=True):
        """Converts a 1xN Tensor into a list of Tensors, each matching the 
        dimension of the corresponding parameter in the model.
        """
        # print("_vec2modelrepr")
        assert self.model.count_parameters(only_trainable=True) == vec.numel()
        # if yield_generator:
        #     i = 0
        #     for p in self.model.parameters():
        #         j = i + p.numel()
        #         yield vec[i:j].view(p.size())
        #         i = j
        # else:
        pars = []
        i = 0
        for p in self.model.parameters():
            j = i + p.numel()
            pars.append(vec[i:j].view(p.size()))
            i = j
        return pars

    def _modelrepr2vec(self, modelrepr, only_trainable=True):
        """Converts a list of Tensors to a single 1xN Tensor.
        """
        if all([p is None for p in modelrepr]):
            return torch.ones(self.model.count_parameters(only_trainable=True))
        else:
            vec = torch.zeros(self.model.count_parameters(only_trainable=True))
            i = 0
            for p in modelrepr:
                j = i + p.numel()
                vec[i:j] = p.data.view(-1) if type(p) is Variable else p.view(-1)
                i = j
            return vec

    def compute_sensitivities(self, inputs=None, do_normalize=True, do_numerical=True):
        """Computes the output-weight sensitivities of the model.
       
        Currently implements the SM-G-ABS and SM-G-SUM safe mutations.

        Reference: Safe Mutations for Deep and Recurrent Neural Networks through Output Gradients [2017]
        
        Args:
            inputs (np.array): Batch of inputs on which to backpropagate and estimate sensitivities
            do_normalize (bool, optional): Defaults to True. Rescale all sensitivities to be between 0 and 1
            do_numerical (bool, optional): Defaults to True. Make sure sensitivities are not numerically ill conditioned (close to zero)
        
        Raises:
            NotImplementedError: For the SO safe mutation
            ValueError: For an unrecognized safe mutation
        """ 
        # Forward pass on input batch
        if inputs is None:
            inputs = self.sens_inputs
        if type(inputs) is not Variable:
            inputs = Variable(inputs)
        if self.cuda:
            inputs = inputs.cuda()
            self.model.cuda()
        if self.safe_mutation is None:
            # Dummy backprop to initialize gradients, then return
            output = self.model(inputs)
            if self.cuda:
                t = torch.cuda.FloatTensor(1, output.data.size()[1]).fill_(0)
            else:
                t = torch.zeros(1, output.data.size()[1])
            t[0, 0] = 1
#            output[0].backward(t) #TODO changed
            output[0].backward(t[0])
            self.model.zero_grad()
            return
        outputs = self.model(inputs)
        batch_size = outputs.data.size()[0]
        n_outputs = outputs.data.size()[1]
        if self.cuda:
            t = torch.cuda.FloatTensor(batch_size, n_outputs).fill_(0)
        else:
            t = torch.zeros(batch_size, n_outputs)
        # Compute sensitivities using specified method
        if self.safe_mutation == 'ABS':
            sensitivities = self._compute_sensitivities_abs(outputs, t)
        elif self.safe_mutation == 'SUM':
            sensitivities = self._compute_sensitivities_sum(outputs, t)
        elif self.safe_mutation == 'SO':
            raise NotImplementedError('The second order safe mutation (SM-SO) is not yet implemented')
        elif self.safe_mutation == 'R':
            raise NotImplementedError('The SM-R safe mutation is not yet implemented')
        else:
            raise ValueError('The type ''{:s}'' of safe mutations is unrecognized'.format(self.safe_mutation))
        # Remove infs
        if do_numerical:
            overflow = False
            for pid in range(len(sensitivities)):
                sensitivities[pid][np.isinf(sensitivities[pid])] = 1
                overflow = overflow or np.isinf(sensitivities[pid]).any()
            if overflow:
                print('| Encountered numerical overflow in sensitivities', end='')
        # Normalize
        if do_normalize:
            # Find maximal sensitivity across all layers
            m = 0
            for pid in range(len(sensitivities)):
                m = np.max([m, sensitivities[pid].max()])
            if m == 0:
                print(' | All sensitivities were zero.', end='')
                for pid in range(len(sensitivities)):
                    sensitivities[pid] = None
                self.sensitivities = sensitivities
                return
            else:
                # Divide all layers by max (and clamp below and above)
                for pid in range(len(sensitivities)):
                    sensitivities[pid] /= m
                    sensitivities[pid].clamp_(min=1e-2, max=1)
        # Set sensitivities and assert their values
        for sens in sensitivities:
            assert not np.isnan(sens).any()
            assert not np.isinf(sens).any()
        self.sensitivities = sensitivities

    def _compute_sensitivities_abs(self, outputs, t):
        # Backward pass for each output unit (and accumulate gradients)
        sensitivities = []
        for k in range(t.size()[1]):
            self.model.zero_grad()
            # Compute dy_t/dw on batch
            for i in range(t.size()[0]):
                t.fill_(0)
                t[i, k] = 1
                outputs.backward(t, retain_graph=True)
                for param in self.model.parameters():
                    param.grad.data = param.grad.data.abs()
            # Get computed sensitivities and sum into those of other output units
            for pid, param in enumerate(self.model.parameters()):
                sens = param.grad.data.clone()  # Clone to sum correctly
                sens = sens.div(t.size()[0]).pow(2)
                if k == 0:
                    sensitivities.append(sens)
                else:
                    sensitivities[pid] += sens
        for pid, _ in enumerate(sensitivities):
            sensitivities[pid] = sensitivities[pid].sqrt()
        return sensitivities

    def _compute_sensitivities_sum(self, outputs, t):
        sensitivities = []
        # Backward pass for each output unit (and accumulate gradients)
        for k in range(t.size()[1]):
            self.model.zero_grad()
            # Compute dy_t/dw on batch
            # Sum over signed gradients
            t.fill_(0)
            t[:, k].fill_(1)
            outputs.backward(t, retain_graph=True)
            # Get computed sensitivities and sum into those of other output units
            for pid, param in enumerate(self.model.parameters()):
                sens = param.grad.data.clone()  # Clone to sum correctly
                sens = sens.pow(2)
                if k == 0:
                    sensitivities.append(sens)
                else:
                    sensitivities[pid] += sens
        for pid, _ in enumerate(sensitivities):
            sensitivities[pid] = sensitivities[pid].sqrt()
        return sensitivities

    def _compute_sensitivities_so(self, outputs, t):
        pass
    
    def _compute_sensitivities_r(self, outputs, t):
        pass

    def generate_sample(self, seed, mean, sigma):
        """Separable case
        """
        sign = np.sign(seed).float()
        torch.manual_seed(abs(seed))
        torch.cuda.manual_seed(abs(seed))
        sample = []
        for p, w, s, sens in zip(self.model.parameters(), mean, sigma, self.sensitivities):
            eps = sign * self.get_perturbation(p.size(), sensitivities=sens, cuda=self.cuda)
            sample.append(w + s * eps)
        # print("Importance mixing")
        # print(type(seed))
        # print(type(sign))
        # print(seed)
        # print(eps) # REMOVE
        return self._modelrepr2vec(sample)
    
    @staticmethod
    def gaussian_log_p(sample, mean, sigma):
        """Separable case. PDF is product of univariate Gaussian PDFs
        """
        log_p = - 0.5 * sample.numel() * np.log(2*np.pi) - 0.5 * sigma.pow(2).prod() - 0.5 * sigma.pow(-2) * (sample - mean) @ (sample - mean)
        # log_p = 0
        # for sample_i, m, s in zip(sample, mean, sigma):
        #     n = torch.distributions.Normal(m, s)
        #     log_p += n.log_prob(sample_i)
        return log_p

    @staticmethod
    def compute_importance_weight(self, sample, m1, s1, m2, s2):
        iw = np.exp(self.gaussian_log_p(sample, m1, s1) - self.gaussian_log_p(sample, m2, s2))
        return iw

    @staticmethod
    def cosine_similarity(v1, v2, dim=0, eps=1e-8):
        w12 = torch.sum(v1 * v2, dim=dim)
        w1 = torch.norm(v1, 2, dim=dim)
        w2 = torch.norm(v2, 2, dim=dim)
        return (w12 / (w1 * w2).clamp(min=eps)).clamp(max=1)

    def compute_pseudo_importance_weight(self, s, m1, s1, m2, s2):
        """Computes the ratio of the cosine similarity of the sample to each of the means.
        
        This functions as a pseudo-importance weight.
        The more similar the sample is to m1, the higher the importance weight
        """
        # Cosine similarity
        cs1 = self.cosine_similarity(s, m1)
        cs2 = self.cosine_similarity(s, m2)
        # Angular distance
        nad1 = cs1.acos()/np.pi
        nad2 = cs2.acos()/np.pi
        # Angular similarity
        nas1 = 1 - nad1
        nas2 = 1 - nad2
        iw = nas1 / nas2
        return iw, nas1, nas2

    def importance_mixing(self, seeds, current_pdf_pars, previous_pdf_pars):
        # Get input
        if self.__class__ in [ES, NES, sNES, sES]:
            prev_mean, prev_sigma = previous_pdf_pars
            curr_mean, curr_sigma = current_pdf_pars
            prev_mean = self._modelrepr2vec(prev_mean)
            curr_mean = self._modelrepr2vec(curr_mean)
            if self.optimize_sigma in [None, 'single']:
                n_weights = self.model.count_parameters()
                prev_sigma = prev_sigma.clone().repeat(n_weights)  # TODO These .clone()s should be possible to remove in pytorch 3.1
                curr_sigma = curr_sigma.clone().repeat(n_weights)
            elif self.optimize_sigma == 'per-layer':
                raise NotImplementedError()
            elif self.optimize_sigma == 'per-weight':
                pass

        # curr_mean = torch.FloatTensor([0.1])
        # curr_sigma = torch.FloatTensor([1])
        # prev_mean = torch.FloatTensor([0])
        # prev_sigma = torch.FloatTensor([1])
        # sample = torch.FloatTensor(1).normal_()
        # importance_weight = np.exp(self.gaussian_log_p(sample, curr_mean, curr_sigma) - self.gaussian_log_p(sample, prev_mean, prev_sigma))

        # Rejection sample on previous population keeping some samples for reuse
        # TODO: If antithetic maybe loop only over positives and always accept/reject pairwise?
        reused_ids = []
        if self.forced_refresh != 1.0:
            for i in range(len(seeds)):
                # Generate sample
                prev_mean, prev_sigma = self._vec2modelrepr(prev_mean), self._vec2modelrepr(prev_sigma)
                sample = self.generate_sample(seeds[i], prev_mean, prev_sigma)
                prev_mean, prev_sigma = self._modelrepr2vec(prev_mean), self._modelrepr2vec(prev_sigma)
                # Reject/accept old sample into new distribution using importance weights
                importance_weight = self.compute_importance_weight(sample, curr_mean, curr_sigma, prev_mean, prev_sigma)
                p_accept = (1 - self.forced_refresh) * importance_weight
                r = np.random.uniform(0, 1)
                if r < p_accept:
                    reused_ids.append(i)
                # Never use only old samples
                if self.perturbations - len(reused_ids) <= max(1, self.perturbations * self.forced_refresh):
                    break
        
        reused_seeds = seeds[reused_ids] if reused_ids else torch.LongTensor([])
        # Reverse rejection sample from new distribution until the wanted total population size is reached
        # always making sure the new population conforms to the new search distribution.
        new_seeds = torch.LongTensor([])
        n_rejected = 0
        while len(reused_seeds) + len(new_seeds) < self.perturbations:
            seed = torch.LongTensor(1).random_()
            curr_mean, curr_sigma = self._vec2modelrepr(curr_mean), self._vec2modelrepr(curr_sigma)
            sample = self.generate_sample(seed[0], curr_mean, curr_sigma)
            curr_mean, curr_sigma = self._modelrepr2vec(curr_mean), self._modelrepr2vec(curr_sigma)
            importance_weight = self.compute_importance_weight(sample, curr_mean, curr_sigma, prev_mean, prev_sigma)
            p_accept = 1 - importance_weight
            r = np.random.uniform(0, 1)
            if r < self.forced_refresh or r < p_accept:
                if not self.no_antithetic: seed = torch.cat([seed, -seed])
                new_seeds = torch.cat([new_seeds, seed])
            else:
                n_rejected += 1
        return new_seeds, reused_seeds, reused_ids, n_rejected

    def adaptation_sampling_learning_rate_update(self, returns, seeds, current_pdf_pars, update_ids):
        def revert_optimization_direction(original_optimizer_state):
            for p in self.model.parameters():
                p.grad.data *= -1
                
        # Get input
        if self.__class__ in [ES, NES, sNES, sES]:
            curr_mean, curr_sigma = current_pdf_pars
            curr_mean = self._modelrepr2vec(curr_mean)
            if self.optimize_sigma in [None, 'single']:
                n_weights = self.model.count_parameters()
                curr_sigma = curr_sigma.repeat(n_weights)
            elif self.optimize_sigma == 'per-layer':
                raise NotImplementedError()
            elif self.optimize_sigma == 'per-weight':
                pass

        # Potential better weights (larger step)
        original_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        new_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        IPython.embed()
        if update_ids == 0:
            # Get potential new mean
            potn_mean = []
            new_optimizer_state['param_groups'][0]['lr'] *= 1.1
            self.optimizer.load_state_dict(new_optimizer_state)
            # for p in self.model.parameters():
            #     a = 1
            # print(p.data)
            self.optimizer.step()
            # for p in self.model.parameters():
            #     a = 1
            # print(p.data)
            for p in self.model.parameters():
                potn_mean.append(p.clone())
            self.optimizer.load_state_dict(new_optimizer_state)
            revert_optimization_direction(new_optimizer_state)
            self.optimizer.step()
            self.optimizer.load_state_dict(original_optimizer_state)
            revert_optimization_direction(original_optimizer_state)
            # for p in self.model.parameters():
            #     a = 1
            # print(p.data)
            # Compute importance weights
            importance_weights = []
            potn_mean = self._modelrepr2vec(potn_mean)
            for i in range(len(seeds)):
                # Generate sample
                potn_mean, curr_sigma = self._vec2modelrepr(potn_mean), self._vec2modelrepr(curr_sigma)
                sample = self.generate_sample(seeds[i], potn_mean, curr_sigma)
                potn_mean, curr_sigma = self._modelrepr2vec(potn_mean), self._modelrepr2vec(curr_sigma)
                # Compute importance weight
                importance_weight = self.compute_pseudo_importance_weight(sample, potn_mean, curr_sigma, curr_mean, curr_sigma)[0]
                # iw = self.compute_importance_weight(sample, potn_mean, curr_sigma, curr_mean, curr_sigma)
                importance_weights.append(importance_weight)
            
            importance_weights = np.array(importance_weights)
            unit_weights = np.array([1] * len(importance_weights))
            So = self.fitness_shaping(returns)
            Sw = self.fitness_shaping(np.array(importance_weights) * returns)
            cp = 0.1
            lr = original_optimizer_state['param_groups'][0]['lr']
            lr_init = original_optimizer_state['param_groups'][0]['initial_lr']
            from scipy.stats import mannwhitneyu as mwt
            mwt(So, Sw)
            # TODO Check implementation of WMWT gives same results as unweighted for unit weights
            # TODO Check if unweighted MWT on So and Sw yields reasonable results
            if self.Mann_Whitney_test(So, So, weights=(unit_weights, importance_weights)) < rho:
                # Decay learning rate
                original_optimizer_state['param_groups'][0]['lr'] = (1 - cp) * lr + cp * lr_init
            else:
                # Increase learning rate
                original_optimizer_state['param_groups'][0]['lr'] *= 1.1
            self.optimizer.load_state_dict(original_optimizer_state)
    
    @staticmethod
    def Mann_Whitney_test(S1, S2, weights=None):
        if weights is None:
            weights = (np.array([1] * len(S1)), np.array([1] * len(S2)))
        w1 = weights[0]
        w2 = weights[1]
        IPython.embed()
        ids_eq = S1 == S2
        ids_larger = S1 > S2
        U = w1[ids_larger].dot(w2[ids_larger]) + 0.5 * w1[ids_eq].dot(w2[ids_eq])
        m1 = w1.sum()
        m2 = w2.sum()
        mu = 0.5 * m1 * m2
        s = np.sqrt(m1 * m2 * (m1 + m2 + 1) / 12)
        return (U - mu) / s

    def print_init(self):
        """Print the initial message when training is started
        """
        # Get strings to print
        env_name = self.env.spec.id if hasattr(self.env, 'spec') else self.env.dataset.root.split('/')[-1]
        safe_mutation = self.safe_mutation if self.safe_mutation is not None else 'None'
        # Build init string
        s = "=================== SYSTEM ====================\n"
        s += "System                {:s}\n".format(platform.system())
        s += "Machine               {:s}\n".format(platform.machine())
        s += "Platform              {:s}\n".format(platform.platform())
        s += "Platform version      {:s}\n".format(platform.version())
        s += "Processor             {:s}\n".format(platform.processor())
        s += "Available CPUs        {:s}\n".format(str(mp.cpu_count()))
        s += "\n==================== MODEL ====================\n"
        s += "Summary of " + self.model.__class__.__name__ + "\n\n"
        pd.set_option('display.max_colwidth', -1)
        s += self.model.summary.to_string() + "\n\n"
        s += "Parameters: {:d}".format(self.model.summary.n_parameters.sum()) + "\n"
        s += "Trainable parameters: {:d}".format(self.model.summary.n_trainable.sum()) + "\n"
        s += "Layers: {:d}".format(self.model.summary.shape[0]) + "\n"
        s += "Trainable layers: {:d}".format((self.model.summary.n_trainable != 0).sum()) + "\n"
        s += "\n================== OPTIMIZER ==================\n"
        s += str(type(self.optimizer)) + "\n"
        s += pprint.pformat(self.optimizer.state_dict()['param_groups']) + "\n"
        s += "\n================= LR SCHEDULE =================\n"
        s += str(type(self.lr_scheduler)) + "\n"
        s += pprint.pformat(vars(self.lr_scheduler)) + "\n"
        s += "\n================== ALGORITHM ==================\n"
        s += "Algorithm             {:s}\n".format(self.__class__.__name__)
        s += "Environment           {:s}\n".format(env_name)
        s += "Perturbations         {:d}\n".format(self.perturbations)
        s += "Generations           {:d}\n".format(self.max_generations)
        s += "Batch size            {:<5d}\n".format(self.batch_size)
        s += "Safe mutation         {:s}\n".format(safe_mutation)
        s += "Antithetic sampling   {:s}\n".format(str(not self.no_antithetic))
        s += "Adaptation sampling   {:s}\n".format(str(self.adaptation_sampling))
        s += "Importance sampling   {:f}\n".format(self.forced_refresh)
        s += "Common random numbers {:s}\n".format(str(self.common_random_numbers))
        s += "Workers               {:d}\n".format(self.workers)
        s += "Validation interval   {:d}\n".format(self.val_every)
        s += "Checkpoint interval   {:d}s\n".format(self.chkpt_int)
        s += "Checkpoint directory  {:s}\n".format(self.chkpt_dir)
        s += "CUDA                  {:s}\n".format(str(self.cuda))
        if self.chkpt_dir is not None:
            with open(os.path.join(self.chkpt_dir, 'init.log'), 'a') as f:
                f.write(s)
        if not self.silent:
            print(s, end='')

    def print_iter(self):
        """Print information on a generation during training.
        """
        if self.silent:
            return
        try:
            # O = 'O {1:' + str(len(str(self.batch_size * self.max_generations * self.perturbations))) + 'd}'
            G = 'G {0:' + str(len(str(self.max_generations))) + 'd}'
            R = 'R {5:' + str(len(str(self.perturbations))) + 'd}'
            s = '\n' + G + ' | F {1:7.2f} | A {2:7.2f} | Ma {3:7.2f} | Mi {4:7.2f} | ' + R + ' | L {6:5.4f}'
            s = s.format(self.stats['generations'][-1], self.stats['return_unp'][-1],
                         self.stats['return_avg'][-1],  self.stats['return_max'][-1],
                         self.stats['return_min'][-1],  self.stats['unp_rank'][-1],
                         self.stats['lr_0'][-1])
            if 'accuracy_unp' in self.stats.keys():
                s += ' | TA {:5.1f}%'.format(self.stats['accuracy_unp'][-1]*100)
            if 'return_val' in self.stats.keys() and self.stats['return_val'][-1] is not None:
                s += ' | VF {:7.2f}'.format(self.stats['return_val'][-1])
            if 'accuracy_val' in self.stats.keys() and self.stats['accuracy_val'][-1] is not None:
                s += ' | VA {:5.1f}%'.format(self.stats['accuracy_val'][-1]*100)
            
            
            ##TODO: ADDED CAST TO PRINT TO COPE WITH NUMPY VERSION BUG
            #G = 'G {0:' + str(len(str(self.max_generations))) + 'd}'
            #R = 'R {5:' + str(len(str(self.perturbations))) + 'd}'
            #s = '\n' + G + ' | F {} | A {} | Ma {} | Mi {} | ' + R + ' | L {}'
            #s = s.format(str(self.stats['generations'][-1]), str(self.stats['return_unp'][-1]),
            #             str(self.stats['return_avg'][-1]),  str(self.stats['return_max'][-1]),
            #             str(self.stats['return_min'][-1]),  str(self.stats['unp_rank'][-1]),
            #             str(self.stats['lr_0'][-1]))
            #if 'accuracy_unp' in self.stats.keys():
            #    s += ' | TA {}%'.format(str(self.stats['accuracy_unp'][-1]*100))
            #if 'return_val' in self.stats.keys() and self.stats['return_val'][-1] is not None:
            #    s += ' | VF {}'.format(str(self.stats['return_val'][-1]))
            #if 'accuracy_val' in self.stats.keys() and self.stats['accuracy_val'][-1] is not None:
            #    s += ' | VA {}%'.format(str(self.stats['accuracy_val'][-1]*100)
            print(s, end='')
        except Exception:
            print('Could not print_iter', end='')

    def load_checkpoint(self, chkpt_dir, load_best=False, load_algorithm=False):
        """Loads a saved checkpoint.
        
        Args:
            chkpt_dir (str): Path to the checkpoint directory
            load_best (bool, optional): Defaults to False. Denotes whether or not to load the best model encountered. Otherwise the latest will be loaded.
        
        Raises:
            IOError: If the loading fails, an IOError exception is raised
        """
        # Get stats
        with open(os.path.join(chkpt_dir, 'stats.csv'), 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            stats = {k: [to_numeric(v)] for k, v in next(reader).items()}
            for record in reader:
                for k, v in record.items():
                    stats[k].append(to_numeric(v))
            stats.pop('')
        # Load state dict files
        algorithm_file = 'state-dict-best-algorithm.pkl' if load_best else 'state-dict-algorithm.pkl'
        model_file = 'state-dict-best-model.pkl' if load_best else 'state-dict-model.pkl'
        optimizer_file = 'state-dict-best-optimizer.pkl' if load_best else 'state-dict-optimizer.pkl'
        algorithm_state_dict = torch.load(os.path.join(chkpt_dir, algorithm_file))
        model_state_dict = torch.load(os.path.join(chkpt_dir, model_file))
        optimizer_state_dict = torch.load(os.path.join(chkpt_dir, optimizer_file))
        # Load state dicts into objects
        if load_algorithm:
            # Load algorithm state
            self.load_state_dict(algorithm_state_dict)
            # Load added parameters
            self._load_added_parameters(stats, optimizer_state_dict)
        self.chkpt_dir = chkpt_dir
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.lr_scheduler.last_epoch = stats['generations'][-1]
        # Set constants
        self._training_start_time = algorithm_state_dict['_training_start_time'] + (time.time() - (stats['walltimes'][-1] + algorithm_state_dict['_training_start_time']))
        self._max_unp_return = np.max(stats['return_unp'])
        # self._max_unp_return = m
        for k in stats.keys(): self.stats[k] = []

    def _load_added_parameters(self, stats, optimizer_state_dict):
        for p in optimizer_state_dict['param_groups']:
            if 'label' in p and p['label'] != 'model_params':
                assert hasattr(self, p['label'])
                assert type(getattr(self, p['label'])) is dict and 'label' in getattr(self, p['label'])
                assert getattr(self, p['label'])['label'] == p['label']
                setattr(self, p['label'], self._add_parameter_to_optimize(getattr(self, p['label']), overwrite=True))
                # if idx.sum() == 1:
                #     # Scalar parameter
                #     value = stats[key_list[idx][0]][-1]
                #     setattr(self, p['label'], torch.Tensor([value]))
                # else:
                #     # Vector parameter
                #     idx = np.array([p['label'] in k for k in key_list])
                #     if idx.any():
                #         value = build_tensor(stats, key_list[idx])
                #         IPython.embed()
                #         _parameter = getattr(self, p['label'])
                #         _parameter['params'][0].data = value
                #         p = _parameter
                #     else:
                #         print('Warning during loading the checkpoint:')
                #         print('An added optimized parameter must be stored in the state dict to enable loading, `{}` was not.'.format(p['label']))
                #         print('The algorithm will be run using the default parameter, if any specified.')

                        # raise ValueError('An added optimized parameter must be stored in the state dict to enable loading, `{}` was not'.format(p['label']))

    def _store_stats(self, workers_out, unperturbed_out, unperturbed_val_out, generation, rank, workers_time, n_reused, n_rejected):
        # Check existence of required keys. Create if not there.
        ks = filter(lambda k: k != 'seed', workers_out.keys())
        add_dict = {}
        for k in ks: add_dict.update({k + suffix: [] for suffix in ['_min', '_max', '_avg', '_var', '_sum', '_unp', '_val']})
        if not set(add_dict.keys()).issubset(set(self.stats.keys())):
            self.stats.update(add_dict)
        # Append data
        self.stats['generations'].append(generation)
        self.stats['walltimes'].append(time.time() - self._training_start_time)
        self.stats['workertimes'].append(workers_time)
        self.stats['unp_rank'].append(rank)
        self.stats['n_reused'].append(n_reused)
        self.stats['n_rejected'].append(n_rejected)
        self.stats['grad_norm'].append(self.model.gradient_norm().numpy())
        self.stats['param_norm'].append(self.model.parameter_norm().numpy() )
        for i, lr in enumerate(self.lr_scheduler.get_lr()):
            self.stats['lr_' + str(i)].append(lr)
        for k, v in workers_out.items():
            if not k in ['seed']:
                self.stats[k + '_min'].append(np.min(v))
                self.stats[k + '_max'].append(np.max(v))
                self.stats[k + '_avg'].append(np.mean(v))
                self.stats[k + '_var'].append(np.var(v))
                self.stats[k + '_sum'].append(np.sum(v))
                self.stats[k + '_unp'].append(unperturbed_out[k])
                self.stats[k + '_val'].append(unperturbed_val_out[k])

    def _dump_stats(self):
        # Store stats as csv file on drive such that self does not grow in size
        # https://stackoverflow.com/questions/23613426/write-dictionary-of-lists-to-a-csv-file
        csvfile_path = os.path.join(self.chkpt_dir, 'stats.csv')
        df = pd.DataFrame(self.stats, index=self.stats['generations'])
        with open(csvfile_path, 'a') as csvfile:
            print_header = os.stat(csvfile_path).st_size == 0
            df.to_csv(csvfile, header=print_header)
        for k in self.stats.keys(): self.stats[k] = []

    def save_checkpoint(self, best_model_stdct=None, best_optimizer_stdct=None, best_algorithm_stdct=None):
        """
        Save a checkpoint of the algorithm.
        
        The checkpoint consists of `self.model` and `self.optimizer` in the latest and best versions along with 
        statistics in the `self.stats` dictionary.

        Args:
            best_model_stdct (dict, optional): Defaults to None. State dictionary of the checkpoint's best model
            best_optimizer_stdct (dict, optional): Defaults to None. State dictionary of the associated optimizer
            best_algorithm_stdct (dict, optional): Defaults to None. State dictionary of the associated algorithm
        """
        if self.chkpt_dir is None:
            return
        # Save stats
        self._dump_stats()
        # Save latest model and optimizer state
        torch.save(self.state_dict(exclude=True), os.path.join(self.chkpt_dir, 'state-dict-algorithm.pkl'))
        torch.save(self.model.state_dict(), os.path.join(self.chkpt_dir, 'state-dict-model.pkl'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.chkpt_dir, 'state-dict-optimizer.pkl'))
        # Save best model
        if best_model_stdct is not None:
            torch.save(self.state_dict(exclude=True), os.path.join(self.chkpt_dir, 'state-dict-best-algorithm.pkl'))
            torch.save(self.model.state_dict(), os.path.join(self.chkpt_dir, 'state-dict-best-model.pkl'))
            torch.save(self.optimizer.state_dict(), os.path.join(self.chkpt_dir, 'state-dict-best-optimizer.pkl'))
        if not self.silent:
            print(' | checkpoint', end='')
        # Currently, learning rate scheduler has no state_dict and cannot be saved. 
        # It can however be restored by setting lr_scheduler.last_epoch = last generation index since
        # this is the only property that has any effect on its functioning.
        # torch.save(lr_scheduler.state_dict(), os.path.join(self.chkpt_dir, 'state-dict-lr-scheduler.pkl'))

    def state_dict(self, exclude=False):
        """Get the state dictionary of the algorithm.

        Args:
            exclude (bool, optional): Defaults to False. Exlude the attributes defined in self.exlude_from_state_dict

        Returns:
            dict: The state dictionary
        """
        if exclude:
            algorithm_state_dict = self.state_dict().copy()
            for k in self.exclude_from_state_dict:
                algorithm_state_dict.pop(k, None)
            return algorithm_state_dict
        else:
            return vars(self)

    def load_state_dict(self, state_dict):
        """Updates the Algorithm state to that of the given state dictionary.

        If the given state dictionary has keys that are not in the specific Algoritm object or
        if the Algorithm object has attributes that are not in the state dictionary, an error is raised
        
        Args:
            state_dict (dict): Dictionary of the attributes of the Algorithm object.
        """
        # Certain attributes should be missing while others must be present
        assert (set(vars(self)) - set(state_dict)) == self.exclude_from_state_dict, 'The loaded state_dict does not correspond to the chosen algorithm'
        for k, v in state_dict.items():
            if k in self.__dict__.keys():
                self.__dict__[k] = v
        if not self.silent:
            print("\n" + self.__class__.__name__ + ' algorithm restored from state dict.')

#%%
class StochasticGradientEstimation(Algorithm):
    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, baseline_mu, baseline_sigma, small_sigma, **kwargs):
        super(StochasticGradientEstimation, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, **kwargs)
        self.baseline_mu = baseline_mu
        self.baseline_sigma = baseline_sigma
        self.small_sigma = small_sigma
        
        self.stats['baseline_mu'] = []
        self.stats['baseline_sigma'] = []
        
    def _store_stats(self, mu_b, sigma_b, *args):
        super(StochasticGradientEstimation, self)._store_stats(*args)
        self.stats['baseline_mu'].append(mu_b.item())
        self.stats['baseline_sigma'].append(sigma_b.item())

        
    @staticmethod
    def fitness_shaping(returns, baseline_mu_rank=None, baseline_sigma_rank=None): #TODO modified to account for baseline shaping
        """Computes the fitness shaped returns.

        Performs the fitness rank transformation used for CMA-ES.
        Reference: Natural Evolution Strategies [2014]
        
        Args:
            returns (np.array): Returns of evaluated perturbed models.
        
        Returns:
            np.array: Shaped returns
        """
        assert type(returns) == np.ndarray
        n = len(returns)
        sorted_indices = np.argsort(-returns)
        u = np.zeros(n)
        u_baseline_mu = None
        u_baseline_sigma = None
        for k in range(n):
            u[sorted_indices[k]] = np.max([0, np.log(n / 2 + 1) - np.log(k + 1)])
        if baseline_mu_rank is not None:
            k = baseline_mu_rank
            u_baseline_mu = np.max([0, np.log(n / 2 + 1) - np.log(k + 1)])
            u_baseline_mu = u_baseline_mu / np.sum(u) - 1 / n
        if baseline_sigma_rank is not None:
            k = baseline_sigma_rank
            u_baseline_sigma = np.max([0, np.log(n / 2 + 1) - np.log(k + 1)])
            u_baseline_sigma = u_baseline_sigma / np.sum(u) - 1 / n
        return (u / np.sum(u) - 1 / n), u_baseline_mu, u_baseline_sigma
        
    def _eval_wrap(self, seed, **kwargs):
        """Get a perturbed model and a copy of the environment and evaluate the perturbation.
        """
        model = self.perturb_model(seed)
        # env = copy.deepcopy(self.env)
        return self.eval_fun(model, self.env, seed, **kwargs)
    
    ########################################
    # BASELINE GET AND CALCULATION METHODS #
    
    def get_baselines(self, returns, seeds, unp_return=0, unp_rank=None):
        mu_b = 0
        mu_b_rank = None
        
        sigma_b = 0
        sigma_b_rank = None
        
        if self.baseline_mu:
            #If small sigma approximation, baseline_mu = f0
            if self.small_sigma:
                mu_b = unp_return    #if yes, f0 = unperturbed_out 
            #Else: estimate baseline_mu
            else:
                mu_b = self.compute_baseline_mu(returns, seeds)
            #Compute f0 rank
            if self.small_sigma:
                mu_b_rank = unp_rank
            else:
                mu_b_rank = self.unperturbed_rank(returns, mu_b)                       
        if not isinstance(mu_b, torch.Tensor):
            mu_b = torch.FloatTensor([mu_b])
        
        if self.baseline_sigma:
            #If small sigma approximation, baseline_mu = f0
            if self.small_sigma:
                #if yes, f0 = unperturbed_out
                sigma_b = unp_return #TODO: do we want to store the baseline as return or transformed rank?            
            #Else: estimate baseline_sigma
            else:
                sigma_b = self.compute_baseline_sigma(returns, seeds)
            #Compute f0 rank
            if self.small_sigma:
                sigma_b_rank = unp_rank
            else:
                sigma_b_rank = self.unperturbed_rank(returns, sigma_b)                       
        if not isinstance(sigma_b, torch.Tensor):
            sigma_b = torch.FloatTensor([sigma_b])
    
        return mu_b, mu_b_rank, sigma_b, sigma_b_rank
        
    def compute_baseline_mu(self, returns, seeds):
        num = 0
        denom = 0
        for i, retrn in enumerate(returns):
            # Set random seed, get antithetic multiplier and return
            sign = np.sign(seeds[i]).float().numpy() #TODO: added recast to numpy 
            torch.manual_seed(abs(seeds[i]))
            for layer, param in enumerate(self.model.parameters()):
                eps = self.get_perturbation(param.size(), sensitivities=self.sensitivities[layer])
                num += float((sign * retrn) * eps.pow(2).sum()) / (self.sigma**2) 
                denom += eps.pow(2).sum() / (self.sigma**2)
        baseline_mu = num/denom #TODO: added cast to float because REASONS
        return baseline_mu 
    
    def compute_baseline_sigma(self, returns, seeds):
        num = 0
        denom = 0
        for i, retrn in enumerate(returns):
            # Set random seed, get antithetic multiplier and return
            sign = np.sign(seeds[i]).float().numpy()
            torch.manual_seed(abs(seeds[i]))
            for layer, param in enumerate(self.model.parameters()):
                eps = self.get_perturbation(param.size(), sensitivities=self.sensitivities[layer])
                num += float((sign * retrn) * (eps.pow(2).sum() -1)**2) / (self.sigma**4) 
                denom += (eps.pow(2).sum() -1)**2 / (self.sigma**4)
        baseline_sigma = num/denom #added cast to float
        return baseline_sigma    
    

    def train(self):
        def draw_seeds(n):
            seeds = torch.LongTensor(n).random_()
            if not self.no_antithetic: seeds = torch.cat([seeds, -seeds])
            return seeds
        
        # Initialize
        # TODO Maybe possible to move to abstract class either in train method or as distinct methods (e.g. init_train)
        self.print_init()
        self.model.train()
        # Initialize variables dependent on restoring
        if not self._training_start_time:
            # If nothing stored then start from scratch
            start_generation = 0
            max_unperturbed_return = -1e8
            self._training_start_time = time.time()
        else:
            # If something stored then start from last iteration
            start_generation = self.lr_scheduler.last_epoch + 1
            max_unperturbed_return = self._max_unp_return
        # Initialize variables independent of state
        if self.workers > 1: pool = mp.Pool(processes=self.workers)
        if self.track_parallel and "DISPLAY" in os.environ: pb = PoolProgress(pool, update_interval=.5, keep_after_done=False, title='Evaluating perturbations')
        best_algorithm_stdct = None
        best_model_stdct = None
        best_optimizer_stdct = None
        last_checkpoint_time = time.time()
        chunksize = self.perturbations // (10 * self.workers) + int(self.perturbations // (10 * self.workers) == 0)
        eval_kwargs = {'max_episode_length': self.batch_size, 'chunksize': chunksize}
        # eval_kwargs_unp = {'max_episode_length': self.batch_size, 'collect_inputs': hasattr(self.env, 'env'), 'chunksize': chunksize}
        eval_kwargs_unp = {'max_episode_length': self.batch_size, 'collect_inputs': hasattr(self.env, 'env') * self.batch_size, 'chunksize': chunksize}

        # Get initial sensitivities if reinforcement learning
        if hasattr(self.env, 'env'):
            unperturbed_out = self.eval_fun(self.model, self.env, 42, **eval_kwargs_unp)
            self.sens_inputs = torch.from_numpy(unperturbed_out['inputs'])

        # Importance sampling initialization
        # current_weights = []
        # for p in self.model.parameters():
        #     current_weights.append(p.data.clone())
        # current_sigma = self._beta2sigma(self.beta.data)
        reused_return = np.array([]).astype('float32')
        reused_seeds = torch.LongTensor()
        n_rejected = 0

        # Start training loop
        for n_generation in range(start_generation, self.max_generations):
            # Compute parent model weight-output sensitivities
            self.compute_sensitivities()

            # Generate random seeds
            if self.forced_refresh == 0:
                # Regular sampling
                seeds = draw_seeds(int(self.perturbations/((not self.no_antithetic) + 1)))
                assert len(seeds) == self.perturbations, 'Number of created seeds is not equal to wanted perturbations'
            elif 'seeds' in locals() and 0.0 < self.forced_refresh < 1.0:
                # Importance mixing
                if 'reused_ids' in locals():
                    rids = list(reused_ids)
                seeds, reused_seeds, reused_ids, n_rejected = self.importance_mixing(seeds, (current_weights, current_sigma), (previous_weights, previous_sigma))
                print(" | RU " + str(len(reused_seeds)), end='')
                if 'rids' in locals():
                    print(' | 2RU ' + str(np.sum(np.array(rids) == np.array(reused_ids))), end='')
                print(" | RJ " + str(n_rejected), end='')
            elif 'seeds' in locals() and isint(self.forced_refresh) and self.forced_refresh >= 0:
                if 'reused_ids' in locals():
                    rids = list(reused_ids)
                # # Percentile
                # percentile = np.percentile(workers_out['return'], 99)
                # reused_ids = np.where(workers_out['return'] > percentile)[0]
                # reused_seeds = seeds[reused_ids.tolist()]
                # reused_return = workers_out['return'][reused_ids]

                # # Better than unperturbed
                # return_unp = unperturbed_out['return']
                # reused_ids = np.where(workers_out['return'] > return_unp)[0]
                # reused_seeds = seeds[reused_ids.tolist()]
                # reused_return = workers_out['return'][reused_ids]

                # # Using percentile of those better than unperturbed
                # return_unp = unperturbed_out['return']
                # percentile = np.percentile(workers_out['return'], 90)
                # reused_ids = np.where(np.logical_and(workers_out['return'] > return_unp, workers_out['return'] > percentile))[0]
                # reused_seeds = seeds[reused_ids.tolist()] if reused_ids.tolist() else torch.LongTensor([])
                # reused_return = workers_out['return'][reused_ids]
                
                # # Using 2%-tile worst and best
                # percentile = np.percentile(workers_out['return'], 98)
                # reused_ids = workers_out['return'] > percentile
                # percentile = np.percentile(workers_out['return'], 2)
                # reused_ids = np.where(np.logical_or(reused_ids, workers_out['return'] < percentile))[0]
                # reused_seeds = seeds[reused_ids.tolist()]
                # reused_return = workers_out['return'][reused_ids]

                # Random reuse
                n_reused = int(self.perturbations) - int(self.forced_refresh)
                reused_ids = np.random.randint(0, self.perturbations, size=n_reused)
                reused_seeds = seeds[reused_ids.tolist()] if reused_ids.tolist() else torch.LongTensor([])
                reused_return = workers_out['return'][reused_ids]

                # Regular sampling of the new seeds
                seeds = draw_seeds(int(self.perturbations/((not self.no_antithetic) + 1)))
                print(" | RU " + str(len(reused_seeds)), end='')
                if 'rids' in locals():
                    print('| 2RU ' + str(np.sum(np.array(rids) == np.array(reused_ids))), end='')
            else:
                # Online updated linear basis function model on samples (i.e. seen network weights) with objective function as target
                if 'seeds' in locals():
                    # Standard (MSE)
                    # w = w + eta * (t_n - w' * phi_n) * phi_n
                    MSE = 0
                    for r, s in zip(workers_out['return'], seeds):
                        network_weights = self.generate_sample(s, self._modelrepr2vec(current_weights), current_sigma)
                        network_weights.apply_(LBF_basis_fcn)
                        network_weights = torch.cat([network_weights, torch.FloatTensor([1])])
                        prediction = LBF_weights @ network_weights
                        LBF_weights += LBF_eta * (r - prediction) * network_weights
                        MSE += (r - prediction) ** 2
                    MSE /= 2
                    print(' | LBF_MSE {:4.2f}'.format(MSE), end='')

                    prediction = LBF_weights @ torch.cat([self._modelrepr2vec(current_weights).apply_(LBF_basis_fcn), torch.FloatTensor([1])])
                    print(' | LBF_Pr {:4.2f}'.format(prediction), end='')

                    # Regularized (MSE)
                    # w = w + eta * (t_n - w' * phi_n) * phi_n + lambda * w

                    n_rejected = 0
                    seeds = torch.LongTensor(int(self.perturbations/((not self.no_antithetic) + 1))).random_()
                    if not self.no_antithetic: seeds = torch.cat([seeds, -seeds])
                    assert len(seeds) == self.perturbations, 'Number of created seeds is not equal to wanted perturbations'

                else:
                    # First time
                    LBF_n_weights = self.model.count_parameters(only_trainable=True) + 1
                    LBF_weights = torch.FloatTensor(LBF_n_weights).normal_()
                    LBF_s = 1
                    LBF_basis_fcn = lambda x: np.exp(- x**2 / 2) # (2 * LBF_s**2))
                    LBF_eta = 1e-7

                    n_rejected = 0
                    seeds = torch.LongTensor(int(self.perturbations/((not self.no_antithetic) + 1))).random_()
                    if not self.no_antithetic: seeds = torch.cat([seeds, -seeds])
                    assert len(seeds) == self.perturbations, 'Number of created seeds is not equal to wanted perturbations'

            # Get master seed for Common Random Numbers
            if self.common_random_numbers:
                eval_kwargs['mseed'] = draw_seeds(1)[0]
                eval_kwargs_unp['mseed'] = eval_kwargs['mseed']
            # unperturbed_out = self.eval_fun(self.model, self.env, 42, **eval_kwargs_unp)

            # Evaluate perturbations
            workers_start_time = time.time()
            if self.workers > 1:
                # Execute all perturbations on the pool of processes
                workers_out = pool.map_async(partial(self._eval_wrap, **eval_kwargs), seeds)
                unperturbed_out = pool.apply_async(self.eval_fun, (self.model, self.env, 42), eval_kwargs_unp)
                if self.val_env is not None and n_generation % self.val_every == 0:
                    model_class = type(self.model)
                    val_model = model_class(self.env.observation_space, self.env.action_space) if hasattr(self.env, 'observation_space') else model_class()
                    val_model.load_state_dict(self.model.state_dict())
                    val_model.zero_grad()
                    val_model.eval()
                    unperturbed_val_out = pool.apply_async(self.eval_fun, (self.model, self.val_env, 42))
                else:
                    unperturbed_val_out = None
                if self.track_parallel and "DISPLAY" in os.environ: pb.track(workers_out)
                workers_out = workers_out.get(timeout=3600)
                unperturbed_out = unperturbed_out.get(timeout=3600)
                if unperturbed_val_out is None:
                    unperturbed_val_out = {k: None for k in unperturbed_out.keys()}
                else:
                    unperturbed_val_out = unperturbed_val_out.get(timeout=3600)
            else:
                # Execute sequentially
                unperturbed_out = self.eval_fun(self.model, self.env, 42, **eval_kwargs_unp)
                if self.val_env is not None and n_generation % self.val_every == 0:
                    unperturbed_val_out = self.eval_fun(self.model, self.val_env, 42)
                else:
                    unperturbed_val_out = {k: None for k in unperturbed_out.keys()}
                workers_out = []
                for s in seeds:
                    workers_out.append(self._eval_wrap(s, **eval_kwargs))
            workers_time = time.time() - workers_start_time
            assert 'return' in unperturbed_out.keys() and 'seed' in unperturbed_out.keys(), "The `eval_fun` must give a return and repass the used seed"
            if hasattr(self.env, 'env'):
                self.sens_inputs = torch.from_numpy(unperturbed_out['inputs'])

            # Invert output from list of dicts to dict of lists
            workers_out = dict(zip(workers_out[0], zip(*[d.values() for d in workers_out])))

            # Recast all outputs as np.ndarrays and seeds as torch.LongTensor
            for k, v in filter(lambda i: i[0] != 'seed', workers_out.items()): workers_out[k] = np.array(v)
            workers_out['seed'] = torch.LongTensor(workers_out['seed'])

            # Append reused seeds and returns (importance mixing)
            assert (seeds == workers_out['seed']).all(), 'The generated seeds must be the same as those returned by workers (plus reused seeds)'
            workers_out['return'] = np.append(workers_out['return'], reused_return)  # Order is important (resampling combined with returns later)
            workers_out['seed'] = torch.cat([workers_out['seed'], reused_seeds])  # Order is important (resampling combined with returns later)
            seeds = torch.cat([seeds, torch.LongTensor(reused_seeds)])  # Order is important (resampling combined with returns later)
            # assert self.perturbations <= len(workers_out['seed']) <= self.perturbations + 1
            
            # Shaping, rank and compute gradients
            rank = self.unperturbed_rank(workers_out['return'], unperturbed_out['return'])
            
            #TODO: introduce baselines computations
            mu_b, mu_b_rank, sigma_b, sigma_b_rank = self.get_baselines(workers_out['return'], workers_out['seed'], unperturbed_out['return'], rank)
     
            shaped_returns, shaped_baseline_mu, shaped_baseline_sigma = self.fitness_shaping(workers_out['return'], mu_b_rank, sigma_b_rank)
        
            #self.compute_gradients(shaped_returns, workers_out['seed'], shaped_baseline_mu, shaped_baseline_sigma)
            self.compute_gradients(shaped_returns, workers_out['seed'])

            # Adaptation sampling
            if type(self.optimizer) == torch.optim.SGD and self.adaptation_sampling:
                self.adaptation_sampling_learning_rate_update(workers_out['return'], seeds, (current_weights, current_sigma), (0))
                self.adaptation_sampling_learning_rate_update(workers_out['return'], seeds, (current_weights, current_sigma), (1))

            # Update the parameters
            self.model.cpu()  # TODO Find out why the optimizer requires model on CPU even with args.cuda = True
            self.optimizer.step()
            if type(self.lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.lr_scheduler.step(unperturbed_out['return'])
            else:
                self.lr_scheduler.step()

            # Update previous and current weights (importance mixing)
            # previous_weights = list(current_weights)
            # previous_sigma = current_sigma.clone()
            # current_weights = []
            # for p in self.model.parameters():
            #     current_weights.append(p.data.clone())
            # current_sigma = self._beta2sigma(self.beta.data)

            # Keep track of best model
            if unperturbed_out['return'] >= max_unperturbed_return:
                best_model_stdct = self.model.state_dict()
                best_optimizer_stdct = self.optimizer.state_dict()
                best_algorithm_stdct = self.state_dict(exclude=True)
                max_unperturbed_return = unperturbed_out['return']

            # Print and checkpoint
            self._store_stats(mu_b, sigma_b, workers_out, unperturbed_out, unperturbed_val_out, n_generation, rank, workers_time, len(reused_seeds), n_rejected) #TODO: [L] added baseline storage
            self.print_iter()
            if last_checkpoint_time < time.time() - self.chkpt_int:
                self.save_checkpoint(best_model_stdct, best_optimizer_stdct, best_algorithm_stdct)
                #plot_stats(os.path.join(self.chkpt_dir, 'stats.csv'), self.chkpt_dir)
                last_checkpoint_time = time.time()
        
        # End training
        self.save_checkpoint(best_model_stdct, best_optimizer_stdct, best_algorithm_stdct)
        plot_stats(os.path.join(self.chkpt_dir, 'stats.csv'), self.chkpt_dir)
        if self.workers > 1:
            pool.close()
            pool.join()
        print("\nTraining done\n\n")

#%%
class ES(StochasticGradientEstimation):
    """Simple regular gradient evolution strategy based on an isotropic Gaussian search distribution.

    The ES algorithm can be derived in the framework of Variational Optimization (VO).
    """
    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, use_naturgrad, optimize_sigma=None, cov_lr=None, **kwargs):
        super(ES, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, **kwargs)
        self.sigma = sigma
        self.optimize_sigma = optimize_sigma

        self.use_naturgrad = use_naturgrad
        
        self.stats['sigma'] = []
        self.stats['beta'] = []
        
        self.estim_mu_var = 0
        self.var_mu_change = 0
        self.estim_sigma_var = 0
        self.var_sigma_change = 0

        self.stats['estim_mu_var'] = []
        self.stats['var_mu_change'] = []
        self.stats['estim_sigma_var'] = []
        self.stats['var_sigma_change'] = []
        
        # Add beta to optimizer and lr_scheduler
        if self.optimize_sigma is not None:
            beta_val = self._sigma2beta(sigma)
            lr = cov_lr if cov_lr else self.lr_scheduler.get_lr()[0]
            beta_par = {'label': '_beta', 'params': Variable(torch.Tensor([beta_val]), requires_grad=True),
                        'lr': lr, 'weight_decay': 0, 'momentum': 0.9, 'dampening': 0.9}
            self._beta = self._add_parameter_to_optimize(beta_par)
        # Add learning rates to stats
        for i, _ in enumerate(self.lr_scheduler.get_lr()):
            self.stats['lr_' + str(i)] = []

    def _store_stats(self, *args):
        super(ES, self)._store_stats(*args)
        if self.optimize_sigma is not None:
            self.stats['sigma'].append(self.sigma.numpy())
            self.stats['beta'].append(self.beta.data[0].numpy())
        else:
            self.stats['sigma'].append(self.sigma)
            self.stats['beta'].append(self.beta.data[0])
        self.stats['estim_mu_var'].append(self.estim_mu_var)
        self.stats['var_mu_change'].append(self.var_mu_change)
        self.stats['estim_sigma_var'].append(self.estim_sigma_var)
        self.stats['var_sigma_change'].append(self.var_sigma_change)

    @property
    def beta(self):
        if self.optimize_sigma:
            assert type(self._beta) is dict and 'params' in self._beta
            beta = self._beta['params'][0]
            self.sigma = self._beta2sigma(beta.data[0])
        else:
            beta = self._sigma2beta(self.sigma)
        return beta
    
    @beta.setter
    def beta(self, beta):
        if self.optimize_sigma:
            self._beta['params'][0].data = beta
        else:
            self._beta = beta
    
    @staticmethod
    def _sigma2beta(sigma):
        return np.log(sigma**2)

    @staticmethod
    def _beta2sigma(beta):
        return np.sqrt(np.exp(beta))
        # return np.sqrt(np.exp(beta)) if type(beta) is np.float64 else (beta.exp()).sqrt().data.numpy()[0]

    def perturb_model(self, seed):
        """Perturbs the main model.
        """
        # Get model class and instantiate new models as copies of parent
        model_class = type(self.model)
        perturbed_model = model_class(self.env.observation_space, self.env.action_space) if hasattr(self.env, 'observation_space') else model_class()
        perturbed_model.load_state_dict(self.model.state_dict())
        perturbed_model.zero_grad()
        # Handle antithetic sampling
        sign = np.sign(seed).float()
        # Set seed and permute by isotropic Gaussian noise
        torch.manual_seed(abs(seed))
        torch.cuda.manual_seed(abs(seed))
        for pp, sens in zip_longest(perturbed_model.parameters(), self.sensitivities):
            eps = self.get_perturbation(pp.size(), sensitivities=sens)
            pp.data += sign * self.sigma * eps
            if np.isnan(pp.data).any():
                print(pp.data)
                print(sens)
                print(eps)
            assert not np.isnan(pp.data).any()
            assert not np.isinf(pp.data).any()
        return perturbed_model

    def weight_gradient(self, retrn, eps):
        return 1 / (self.perturbations * self.sigma) * (retrn * eps)

    def beta_gradient(self, retrn, eps):
        return 1 / (2 * self.perturbations) * retrn * (eps.pow(2).sum() - eps.numel())

    def compute_gradients(self, returns, seeds, baseline_mu=None, baseline_sigma=None):
        """Computes the gradients of the weights of the model wrt. to the return. 
        
        The gradients will point in the direction of change in the weights resulting in a
        decrease in the return.
        
        Also compute estimator variance and (if using baseline) change in var using it

        """
        ## Indpendent parameter groups sampling
        # Preallocate list with gradients
        weight_gradients = []
        beta_gradient = torch.zeros(1)
        for param in self.model.parameters():
            weight_gradients.append(torch.zeros(param.data.size()))
        self.estim_mu_var = 0
        self.var_mu_change = 0
        self.estim_sigma_var = 0
        self.var_sigma_change = 0
        
        mu_var_t1 = 0
        mu_var_t2 = 0
        sigma_var_t1 = 0
        sigma_var_t2 = 0
        
        var_mu_num = 0
        var_mu_denom = 1
        var_sigma_num = 0
        var_sigma_denom = 1
        
        # Compute gradients
        for i, retrn in enumerate(returns):
            # Set random seed, get antithetic multiplier and return
            sign = np.sign(seeds[i]).float()
            torch.manual_seed(abs(seeds[i]))
            for layer, param in enumerate(self.model.parameters()):
                eps = self.get_perturbation(param.size(), sensitivities=self.sensitivities[layer])
                
                l_mu = (eps / self.sigma).numpy() #store the log-likelihood
                ll_mu = (eps.pow(2).sum() / self.sigma**2).numpy()# store l*l^T #TODO adapt for separable sigmas
                l_sigma = ((eps.pow(2).sum() - 1)/ self.sigma**2).numpy()
                ll_sigma = (l_sigma**2)    
                
                mu_var_t1 += (1 / self.perturbations) * (retrn**2 * ll_mu)
                mu_var_t2 += (1 / self.perturbations) * sum((retrn * l_mu.flatten()))
                
                sigma_var_t1 += (1 / self.perturbations) * (retrn**2 * ll_sigma)
                sigma_var_t2 += (1 / self.perturbations) * (retrn * l_sigma)
                
                #self.estim_mu_var += (1 / self.perturbations) * (retrn**2 * ll_mu - sum((retrn * l_mu.flatten())**2))#TODO   
                #self.estim_sigma_var += (1 / self.perturbations) * (retrn**2 * ll_sigma  - (retrn * l_sigma)**2)
                if baseline_mu is not None:
                    weight_gradients[layer] += self.weight_gradient(sign * (retrn - baseline_mu), eps)
                    var_mu_num += (1 / self.perturbations) * retrn*ll_mu
                    var_mu_denom += (1 / self.perturbations) * ll_mu
                    #self.var_mu_change += (1 / self.perturbations) *  2 * (retrn*ll_mu)**2 / ll_mu
                else:
                    weight_gradients[layer] += self.weight_gradient(sign * retrn, eps)                  
                if self.optimize_sigma is not None:
                    if baseline_sigma is not None:
                        beta_gradient += self.beta_gradient((retrn - baseline_sigma), eps).float() #TODO: should I betatransform the baseline?
                        var_sigma_num += (1 / self.perturbations) * retrn*ll_sigma
                        var_sigma_denom += (1 / self.perturbations) * ll_sigma
                        #self.var_sigma_change += (1 / self.perturbations) * 2 * (retrn*ll_sigma)**2 / ll_sigma
                    else:
                        beta_gradient += self.beta_gradient(retrn, eps).float()
                    
        self.estim_mu_var = mu_var_t1 - mu_var_t2**2
        self.estim_sigma_var = sigma_var_t1 - sigma_var_t2**2        
        if baseline_mu is not None:
            self.var_mu_change = 2 * (var_mu_num**2) / var_mu_denom
        if baseline_sigma is not None:
            self.var_sigma_change = 2 * (var_sigma_num**2) / var_sigma_denom
        
        # Set gradients
        self.optimizer.zero_grad()
        for layer, param in enumerate(self.model.parameters()):
            #print(layer)
            #print(param.grad)
            #print(weight_gradients[layer])
            #param.grad.data = - weight_gradients[layer]
            if self.use_naturgrad:
                param.grad.data = - self.sigma * weight_gradients[layer]
            else:
                param.grad.data = - weight_gradients[layer]
            assert not np.isnan(param.grad.data).any()
            assert not np.isinf(param.grad.data).any()
        if self.optimize_sigma:
            if self.use_naturgrad:
                self.beta.grad = - self.sigma**2 * Variable(beta_gradient, requires_grad=True)
            else:
                self.beta.grad = - Variable(beta_gradient, requires_grad=True)
            assert not np.isnan(self.beta.grad.data).any()
            assert not np.isinf(self.beta.grad.data).any()

    def print_init(self):
        super(ES, self).print_init()
        s = "Sigma                 {:5.4f}\n".format(self.sigma)
        s += "Optimizing sigma      {:s}\n".format(str(self.optimize_sigma))
        s += "Use MU baseline       {:s}\n".format(str(self.baseline_mu))
        s += "Use SIGMA baseline    {:s}\n".format(str(self.baseline_sigma))
        s += "Use natural gradient  {:s}\n".format(str(self.use_naturgrad))
        if self.chkpt_dir is not None:
            with open(os.path.join(self.chkpt_dir, 'init.log'), 'a') as f:
                f.write(s + "\n\n")
        s += "\n=================== Running ===================\n"
        print(s)

    def print_iter(self):
        super(ES, self).print_iter()
        s = " | Sig {:5.4f}".format(self.stats['sigma'][-1])
        print(s, end='')


class NES(ES):
    """Simple natural gradient evolution strategy based on an isotropic Gaussian search distribution
    """
    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, optimize_sigma=None, cov_lr=None, **kwargs):
        super(NES, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, optimize_sigma=optimize_sigma, cov_lr=cov_lr, **kwargs)
        # Scale to enable use of same learning rates as for ES
        self._weight_update_scale = 1/self.sigma**2
        # self._beta_update_scale = 1/(2*self.sigma**4)

    def weight_gradient(self, retrn, eps):
        # Equal to sigma^2 * [regular gradient] (1 / (self.perturbations * self.sigma) * (retrn * eps))
        # To rescale gradients to same size as for ES, we multiply by a factor of 1/sigma_0**2 where
        # sigma_0 is the initial sigma value
        return self._weight_update_scale * (self.sigma / self.perturbations) * (retrn * eps)

    def beta_gradient(self, retrn, eps):
        # Equal to 1/2 * [regular gradient] (1 / (2 * self.perturbations) * retrn * (eps.pow(2).sum() - eps.numel()))
        # Since the natural gradient is independent of beta, there is no scaling required.
        return retrn * (eps.pow(2).sum() - eps.numel()) / (2 * self.perturbations)


class sES(StochasticGradientEstimation):
    """Simple Seperable Evolution Strategy

    An algorithm based on an Evolution Strategy (ES) using a Gaussian search distribution.
    The ES algorithm can be derived in the framework of Variational Optimization (VO).
    """

    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, use_naturgrad, optimize_sigma=None, cov_lr=None, **kwargs):
        super(sES, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, **kwargs)
        assert optimize_sigma in [None, 'single', 'per-layer', 'per-weight'], "Expected `self.optimize_sigma` to be one of [None, 'single', 'per-layer', 'per-weight'] but was {}".format(self.optimize_sigma)
        self.optimize_sigma = optimize_sigma
        # Add sigma to stats
        if self.optimize_sigma == 'per-weight':
            sigma_keys = ['sigma_avg', 'sigma_min', 'sigma_max', 'sigma_med', 'sigma_std']
            beta_keys = ['beta_avg', 'beta_min', 'beta_max', 'beta_med', 'beta_std']
        elif self.optimize_sigma == 'per-layer':
            n = self.model.count_tensors(only_trainable=True)
            sigma_keys = ['sigma_' + str(i) for i in range(n)]
            beta_keys = ['beta_' + str(i) for i in range(n)]
        else:
            sigma_keys = ['sigma']
            beta_keys = ['beta']
        for sk, bk in zip(sigma_keys, beta_keys):
            self.stats[sk] = []
            self.stats[bk] = []
        # Add sigma to optimizer and lr_scheduler
        beta = self._sigma2beta(sigma)
        if self.optimize_sigma is None:
            self._beta = Variable(torch.Tensor([beta]), requires_grad=False)
        else:
            if self.optimize_sigma == 'single':
                assert not isinstance(sigma, (collections.Sequence, np.ndarray, torch.Tensor))
                beta_tensor = torch.Tensor([beta])
            elif self.optimize_sigma == 'per-layer':
                if not isinstance(sigma, (collections.Sequence, np.ndarray, torch.Tensor)):
                    beta_tensor = torch.Tensor([beta] * self.model.count_tensors(only_trainable=True))
                elif isinstance(sigma, (collections.Sequence, np.ndarray)):
                    beta_tensor = torch.Tensor([beta])
                elif isinstance(sigma, torch.Tensor):
                    beta_tensor = beta
            elif self.optimize_sigma == 'per-weight':
                if not isinstance(sigma, (collections.Sequence, np.ndarray, torch.Tensor)):
                    beta_tensor = torch.Tensor([beta] * self.model.count_parameters(only_trainable=True))
                elif isinstance(sigma, (collections.Sequence, np.ndarray)):
                    beta_tensor = torch.Tensor([beta])
                elif isinstance(sigma, torch.Tensor):
                    beta_tensor = beta
            if cov_lr:
                lr = cov_lr
            else:
                lr = self.lr_scheduler.get_lr()[0]
            beta_par = {'label': '_beta', 'params': Variable(beta_tensor, requires_grad=True),
                        'lr': lr, 'weight_decay': 0, 'momentum': 0.9, 'dampening': 0.9}
            self._beta = self._add_parameter_to_optimize(beta_par)
        self.sigma = self._beta2sigma(self.beta.data)
        # Add learning rates to stats
        for i, _ in enumerate(self.lr_scheduler.get_lr()):
            self.stats['lr_' + str(i)] = []

    def _store_stats(self, *args):
        super(sES, self)._store_stats(*args)
        if self.optimize_sigma is not None:
            self.stats['sigma'].append(self.sigma.numpy())
            self.stats['beta'].append(self.beta.data[0].numpy())
        else:
            self.stats['sigma'].append(self.sigma)
            self.stats['beta'].append(self.beta.data[0])

    @property
    def beta(self):
        if self.optimize_sigma:
            assert type(self._beta) is dict and 'params' in self._beta
            beta = self._beta['params'][0]
            self.sigma = self._beta2sigma(beta.data)
        else:
            beta = self._beta
        return beta
    
    @beta.setter
    def beta(self, beta):
        if self.optimize_sigma:
            self._beta['params'][0].data = beta
        else:
            self._beta = beta
    
    @staticmethod
    def _sigma2beta(sigma):
        return np.log(sigma**2)

    @staticmethod
    def _beta2sigma(beta):
        return np.sqrt(np.exp(beta))
    
    def perturb_model(self, seed):
        """Perturbs the main model.
        """
        # Get model class and instantiate new models as copies of parent
        model_class = type(self.model)
        perturbed_model = model_class(self.env.observation_space, self.env.action_space) if hasattr(self.env, 'observation_space') else model_class()
        perturbed_model.load_state_dict(self.model.state_dict())
        perturbed_model.zero_grad()
        # Handle antithetic sampling
        sign = np.sign(seed).float()
        # Set seed and permute by isotropic Gaussian noise
        torch.manual_seed(abs(seed))
        torch.cuda.manual_seed(abs(seed))
        if self.optimize_sigma in [None, 'single']:
            for p, pp, sens in zip_longest(self.model.parameters(), perturbed_model.parameters(), self.sensitivities):
                eps = sign * self.get_perturbation(p.size(), sensitivities=sens)
                pp.data += self.sigma * eps
            # print("Perturb model")
            # print(type(seed))
            # print(type(sign))
            # print(seed)
            # print(eps) # REMOVE
        elif self.optimize_sigma == 'per-layer':
            for layer, (p, pp, sens) in enumerate(zip_longest(self.model.parameters(), perturbed_model.parameters(), self.sensitivities)):
                eps = self.get_perturbation(p.size(), sensitivities=sens)
                pp.data += sign * self.sigma[layer] * eps
        elif self.optimize_sigma == 'per-weight':
            i = 0
            for layer, (p, pp, sens) in enumerate(zip_longest(self.model.parameters(), perturbed_model.parameters(), self.sensitivities)):
                j = i + p.numel()
                eps = self.get_perturbation(p.size(), sensitivities=sens)
                pp.data += sign * (self.sigma[i:j] * eps.view(-1)).view(p.size())
                i = j
        # Check numerical values
        for pp in perturbed_model.parameters():
            assert not np.isnan(pp.data).any()
            assert not np.isinf(pp.data).any()
        return perturbed_model

    def compute_gradients(self, returns, seeds):
        """Computes the gradients of the weights of the model wrt. to the return. 
        
        The gradients will point in the direction of change in the weights resulting in a
        decrease in the return.
        """
        # CUDA
        if self.cuda:
            self.model.cuda()
        
        # Preallocate list with gradients
        weight_gradients = []
        for param in self.model.parameters():
            weight_gradients.append(torch.zeros(param.data.size()))
        beta_gradients = torch.zeros(self.beta.size())

        # Compute gradients
        for i, retrn in enumerate(returns):
            # Set random seed, get antithetic multiplier and return
            sign = np.sign(seeds[i]).float()
            torch.manual_seed(abs(seeds[i]))
            torch.cuda.manual_seed(abs(seeds[i]))
            i = 0
            for layer, param in enumerate(self.model.parameters()):
                eps = self.get_perturbation(param.size(), sensitivities=self.sensitivities[layer], cuda=self.cuda)
                if self.use_naurgrad: #TODO: added Natural Gradient adjustement. TOCHECK if same gradient is computed us lr=1              
                    if not self.optimize_sigma:
                        weight_gradients[layer] += (self.sigma / (self.perturbations)) * (sign * retrn * eps)
                    if self.optimize_sigma == 'single':
                        weight_gradients[layer] += (self.sigma / (self.perturbations)) * (sign * retrn * eps)
                        beta_gradients += 1 / ( 2 * self.perturbations) * retrn * (eps.pow(2).sum() - eps.numel())
                    elif self.optimize_sigma == 'per-layer':
                        weight_gradients[layer] += (self.sigma[layer] / (self.perturbations)) * (sign * retrn * eps)
                        beta_gradients[layer] += 1 / ( 2 * self.perturbations) * retrn * (eps.pow(2).sum() - eps.numel())
                    elif self.optimize_sigma == 'per-weight':
                        j = i + param.numel()
                        weight_gradients[layer] += (self.sigma[i:j].view(weight_gradients[layer].size()) / (self.perturbations)) * (sign * retrn * eps)
                        beta_gradients[i:j] += 1 / ( 2 * self.perturbations) * retrn * (eps.view(-1).pow(2) - 1)
                        i = j
                #TODO: need to check naturgrad direction calculation taking into account the betatransform
                else:
                    if not self.optimize_sigma:
                        weight_gradients[layer] += (1 / (self.perturbations * self.sigma)) * (sign * retrn * eps)
                    if self.optimize_sigma == 'single':
                        weight_gradients[layer] += (1 / (self.perturbations * self.sigma)) * (sign * retrn * eps)
                        beta_gradients += (2*np.exp(2*self.beta)) / (self.perturbations) * retrn * (eps.pow(2).sum() - eps.numel())
                    elif self.optimize_sigma == 'per-layer':
                        weight_gradients[layer] += (1 / (self.perturbations * self.sigma[layer])) * (sign * retrn * eps)
                        beta_gradients[layer] += (2*np.exp(2*self.beta[layer])) / (self.perturbations) * retrn * (eps.pow(2).sum() - eps.numel())
                    elif self.optimize_sigma == 'per-weight':
                        j = i + param.numel()
                        weight_gradients[layer] += (1 / (self.perturbations * self.sigma[i:j].view(weight_gradients[layer].size()))) * (sign * retrn * eps)
                        beta_gradients[i:j] += (2*np.exp(2*self.beta[i:j].view(weight_gradients[layer].size()))) / (self.perturbations) * retrn * (eps.view(-1).pow(2) - 1)
                        i = j
  
        # Set gradients
        self.optimizer.zero_grad()
        for layer, param in enumerate(self.model.parameters()):
            param.grad.data = - weight_gradients[layer]
            assert not np.isnan(param.grad.data).any()
            assert not np.isinf(param.grad.data).any()
        if self.optimize_sigma:
            self.beta.grad = - Variable(beta_gradients, requires_grad=True)
            assert not np.isnan(self.beta.grad.data).any()
            assert not np.isinf(self.beta.grad.data).any()

    def print_init(self):
        super(sES, self).print_init()
        s = "Optimizing sigma      {:s}\n".format(str(self.optimize_sigma))
        if self.optimize_sigma in [None, 'single']:
            s += "Sigma                 {:5.4f}\n".format(self.sigma[0])
        elif self.optimize_sigma in ['per-layer', 'per-weight']:
            s += "Sigma mean            {:5.4f}\n".format(self.sigma.mean())
        if self.chkpt_dir is not None:
            with open(os.path.join(self.chkpt_dir, 'init.log'), 'a') as f:
                f.write(s + "\n\n")
        s += "\n=================== Running ===================\n"
        print(s)

    def print_iter(self):
        super(sES, self).print_iter()
        if self.optimize_sigma == 'single':
            s = " | Sig {:5.4f}".format(self.sigma[0])
            print(s, end='')
        elif self.optimize_sigma in ['per-layer', 'per-weight']:
            s = " | Sig {:5.4f}".format(self.sigma.mean())
            print(s, end='')

    def _store_stats(self, *args):
        super(sES, self)._store_stats(*args)
        if (self.optimize_sigma is None) or (self.optimize_sigma == 'single'):
            self.stats['sigma'].append(self.sigma.view(-1)[0].numpy())
            self.stats['beta'].append(self.beta.data.view(-1)[0].numpy())
        elif self.optimize_sigma == 'per-layer':
            for i, s in enumerate(self.sigma):
                self.stats['sigma_' + str(i)].append(s.numpy())
            for i, b in enumerate(self.beta.data):
                self.stats['beta_' + str(i)].append(b.numpy())
        elif self.optimize_sigma == 'per-weight':
            # Compute mean, min, max, median and std
            self.stats['sigma_avg'].append(self.sigma.mean().numpy())
            self.stats['sigma_min'].append(self.sigma.min().numpy())
            self.stats['sigma_max'].append(self.sigma.max().numpy())
            self.stats['sigma_med'].append(self.sigma.median().numpy())
            self.stats['sigma_std'].append(self.sigma.std().numpy())
            # Avoid the recomputation for beta
            self.stats['beta_avg'].append(self._sigma2beta(self.stats['sigma_avg'][-1]))
            self.stats['beta_min'].append(self._sigma2beta(self.stats['sigma_min'][-1]))
            self.stats['beta_max'].append(self._sigma2beta(self.stats['sigma_max'][-1]))
            self.stats['beta_med'].append(self._sigma2beta(self.stats['sigma_med'][-1]))
            self.stats['beta_std'].append(self.beta.data.std().numpy())


class sNES(sES):
    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, optimize_sigma=None, **kwargs):
        super(sNES, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, optimize_sigma=optimize_sigma, **kwargs)
        # Gradient updates scales to make gradient sizes similar to those using regular gradients
        if self.optimize_sigma == 'single':
            self._weight_update_scale = 1/self.sigma**2
            # self._sigma_update_scale = 1/(2*self.sigma**4)
        else:
            self._weight_update_scale = 1/self.sigma.mean()**2
            # self._sigma_update_scale = 1/(2*self.sigma.mean()**4)

    def compute_gradients(self, returns, seeds):
        """Computes the gradients of the weights of the model wrt. to the return. 
        
        The gradients will point in the direction of change in the weights resulting in a
        decrease in the return.
        """
        # CUDA
        if self.cuda:
            self.model.cuda()
        
        # Preallocate list with gradients
        weight_gradients = []
        for param in self.model.parameters():
            weight_gradients.append(torch.zeros(param.data.size()))
        beta_gradients = torch.zeros(self.beta.size())

        # Compute gradients
        for i, retrn in enumerate(returns):
            # Set random seed, get antithetic multiplier and return
            sign = np.sign(seeds[i]).float()
            torch.manual_seed(abs(seeds[i]))
            torch.cuda.manual_seed(abs(seeds[i]))
            j = 0 
            for layer, param in enumerate(self.model.parameters()):
                eps = self.get_perturbation(param.size(), sensitivities=self.sensitivities[layer], cuda=self.cuda)
                if not self.optimize_sigma:
                    weight_gradients[layer] += self._weight_update_scale * self.sigma * (sign * retrn * eps) / self.perturbations
                if self.optimize_sigma == 'single':
                    weight_gradients[layer] += self._weight_update_scale * self.sigma * (sign * retrn * eps) / self.perturbations
                    beta_gradients += float(retrn * (eps.pow(2).sum() - eps.numel()))/ self.perturbations
                elif self.optimize_sigma == 'per-layer':
                    weight_gradients[layer] += self._weight_update_scale * self.sigma[layer] * (sign * retrn * eps) / self.perturbations
                    beta_gradients[layer] += float(retrn * (eps.pow(2).sum() - eps.numel())) / self.perturbations #TODO: added float cast because REASONS
                elif self.optimize_sigma == 'per-weight':
                    k = j + param.numel()
                    weight_gradients[layer] += self._weight_update_scale * self.sigma[j:k].view(weight_gradients[layer].size()) * (sign * retrn * eps) / self.perturbations
                    beta_gradients[j:k] += retrn * (eps.view(-1).pow(2) - 1) / self.perturbations
                    j = k

        # Set gradients
        self.optimizer.zero_grad()
        for layer, param in enumerate(self.model.parameters()):
            param.grad.data = - weight_gradients[layer]
            assert not np.isnan(param.grad.data).any()
            assert not np.isinf(param.grad.data).any()
        if self.optimize_sigma:
            self.beta.grad = - Variable(beta_gradients, requires_grad=True)
            assert not np.isnan(self.beta.grad.data).any()
            assert not np.isinf(self.beta.grad.data).any()


class Backprop(Algorithm):
    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, **kwargs):
        self.train_loader = self.env
        self.test_loader = self.val_env
        if self.cuda:
            self.model.cuda()
        # TODO Add storing of statistics in stats

    def train(self):
        def train_epoch(epoch):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()
                # if batch_idx % args.log_interval == 0:
                #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch, batch_idx * len(data), len(self.train_loader.dataset),
                #         100. * batch_idx / len(self.train_loader), loss.data[0]))
            
        for epoch in range(1, self.max_generations + 1):
            self.test()
            train_epoch(epoch)
            torch.save(self.model.state_dict(), 'latest.state')

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # test_loss /= len(self.test_loader.dataset)
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(self.test_loader.dataset),
        #     100. * correct / len(self.test_loader.dataset)))


class xNES(StochasticGradientEstimation):
    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, sigma, optimize_sigma=None, **kwargs):
        super(xNES, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, **kwargs)
        assert optimize_sigma in ['per-layer', 'per-weight']
        self.optimize_sigma = optimize_sigma
        # Form Sigma as matrix if not already
        if type(sigma) in [float, int]:
            if self.optimize_sigma == 'per-layer':
                Sigma = torch.diag(torch.Tensor([sigma] * self.model.count_tensors(only_trainable=True)))
            elif self.optimize_sigma == 'per-weight':
                Sigma = torch.diag(torch.Tensor([sigma] * self.model.count_parameters(only_trainable=True)))
        else:
            assert type(sigma) is torch.Tensor and \
                   (self.optimize_sigma == 'per-layer' and sigma.numel() == self.model.count_tensors(only_trainable=True)) or \
                   (self.optimize_sigma == 'per-weight' and sigma.numel() == self.model.count_parameters(only_trainable=True))
            Sigma = sigma
        # Decouple shape (B) and scale (sigma) information
        # Get dimensionality of search distribution (Compute 'dimensionality' of problem from perturbations np.exp((self.perturbations - 4) / 3))
        self.d = Sigma.size()[0]
        # Cholesky factorize Sigma
        A = torch.potrf(Sigma)
        # Compute scale: Compute the d'th root of the determinant of A
        self.sigma = np.abs(np.linalg.det(A.numpy())) ** (1 / self.d)
        self.sigma = Variable(torch.Tensor([self.sigma]), requires_grad=True)
        # Compute shape: Normalize cholesky factor of initial Sigma by scale
        self.B = Variable(A/self.sigma.data, requires_grad=True)

        # Find maximal number of elements in any model tensor
        m = 0
        for p in self.model.parameters():
            if p.requires_grad:
                m = int(np.max([m, p.numel()]))
        self.model.max_elements_in_a_tensor = m

        self.compute_sensitivities()
        self.perturb_model(42)

    # @staticmethod
    # def get_perturbation(size, sensitivities=None, cuda=False):
    #     """Draws a perturbation tensor of dimension `size` from a standard normal.

    #     If `sensitivities is not None`, then the perturbation is scaled by these.
    #     If sensitivities are given and `sensitivities.size() == size[1:]` then the `size[0]` is 
    #     intepreted as a number of samples each of which is to be scaled by the given sensitivities.
    #     """
    #     if type(size) in [tuple, list]:
    #         size = torch.Size(size)
    #     elif type(size) in [torch.Size, int]:
    #         pass
    #     else:
    #         raise TypeError("Input `size` must be of type `int`, `list`, `tuple` or `torch.Size` but was `{}`".format(type(size).__name__))
    #     if sensitivities is not None and sensitivities.size() == size[1:]:
    #         samples = size[0]
    #     assert sensitivities is None or sensitivities.size() == size or sensitivities.size() == size[1:], "Sensitivities must match size of perturbation"
    #     if cuda:
    #         eps = torch.cuda.FloatTensor(size)
    #     else:
    #         eps = torch.FloatTensor(size)
    #     eps.normal_(mean=0, std=1)
    #     if sensitivities is not None:
    #         if sensitivities.size() == size[1:]:
    #             for s in range(samples):
    #                 eps[s, ...] /= sensitivities      # Scale by sensitivities
    #                 if eps.numel() > 1:
    #                     eps /= eps.std()              # Rescale to unit variance
    #         else:
    #             eps /= sensitivities     # Scale by sensitivities
    #             if eps.numel() > 1:
    #                 eps /= eps.std()     # Rescale to unit variance
    #     return eps

    def perturb_model(self, seed):
        # Get model class and instantiate new models as copies of parent
        model_class = type(self.model)
        perturbed_model = model_class(self.env.observation_space, self.env.action_space) if hasattr(self.env, 'observation_space') else model_class()
        perturbed_model.load_state_dict(self.model.state_dict())
        perturbed_model.zero_grad()
        # Handle antithetic sampling
        sign = np.sign(seed).float()
        # Set seed and permute by isotropic Gaussian noise
        torch.manual_seed(abs(seed))
        torch.cuda.manual_seed(abs(seed))
        if self.optimize_sigma == 'per-layer':
            # Sample a tensor of perturbations. A perturbation vector for each layer is in each row (column).
            # The challenge is that not all layers have the same number of perturbations - pad with None, zeros? Alternatives?
            # How to do this efficiently?

            # Max number of elements in a parameter
            m = self.model.max_elements_in_a_tensor()
            l = self.model.count_tensors(only_trainable=True)
            # Get perturbations from search distribution
            eps = self.get_perturbation(torch.Size((l, m)))
            eps = sign * self.sigma.data * self.B.data.transpose(0, 1) @ eps
            # Perturb model parameters
            for l, (pp, s) in enumerate(zip(perturbed_model.parameters(), self.sensitivities)):
                # Get needed number of perturbations
                e = eps[l, :pp.numel()].view(pp.size())
                # print(e.mean(), e.std())
                e = self.scale_by_sensitivities(e, s)
                # print(e.mean(), e.std())
                pp.data += e
                assert not np.isnan(pp.data).any()
                assert not np.isinf(pp.data).any()

            # IPython.embed()

            # plt.figure()
            # plt.hist(eps[0:1,:])
            # plt.savefig('dist.pdf')

            # plt.figure()
            # plt.plot(eps[0,:].numpy(), eps[1,:].numpy(), 'o')
            # plt.savefig('scatter.pdf')
            
            # import matplotlib.pyplot as plt
            
            # B = [[1, 0.4], [0.4, 2]]
            # self.B = Variable(torch.Tensor(B))
            # plt.figure()
            # plt.plot(saved[0,:].numpy(), saved[1,:].numpy(), 'o')
            # plt.savefig('scatter.pdf')

            # eps = torch.zeros(torch.Size([l, m]))
            # for l, (p, s) in enumerate(zip(self.model.parameters(), self.sensitivities)):
            #     IPython.embed()
            #     eps[l, :] = self.get_perturbation(sample_size, sensitivities=s).view(-1)
                # rem = m % p.numel()
                # n_samples = int(m/p.numel()) + int(rem != 0)
                # sample_size = torch.Size([n_samples, *p.size()])
                # if rem == 0:
                #     eps[l, :] = self.get_perturbation(sample_size, sensitivities=s).view(-1)
                # else:
                #     eps[l, :-rem] = self.get_perturbation(sample_size, sensitivities=s).view(-1)
            
            # for e in eps: 
                # e = self.

        elif self.optimize_sigma == 'per-weight':
            # Sample a single vector of perturbations (local coordinates)
            sensitivities = self._modelrepr2vec(self.sensitivities)
            n_perturbs = self.model.count_parameters(only_trainable=True)
            eps = self.get_perturbation(torch.Size([n_perturbs]), sensitivities=sensitivities)
            # Compute the task coordinates and reshape into list like model
            eps = sign * self.sigma * self.B.data.transpose(0, 1) * eps
            eps = self._vec2modelrepr(eps)
            # Permute each model parameter
            for pp, e in zip(perturbed_model.parameters(), eps):
                pp.data += e
                assert not np.isnan(p.data).any()
                assert not np.isinf(p.data).any()
        return perturbed_model

    def compute_gradients(self, returns, seeds):
        """Computes the gradients of the weights of the model wrt. to the return. 
        
        The gradients will point in the direction of change in the weights resulting in a
        decrease in the return.
        """
        # CUDA
        if self.cuda:
            self.model.cuda()

        ## Indpendent parameter groups sampling
        # Preallocate list with gradients
        weight_gradients = []
        beta_gradient = 0
        for param in self.model.parameters():
            weight_gradients.append(torch.zeros(param.data.size()))

        # Dependent parameter groups sampling (requires more memory)
        # Preallocate weight gradients as 1xn vector where n is number of parameters in model
        print("xNES compute gradients")
        IPython.embed()
        delta_gradients = torch.zeros(self.model.count_parameters(only_trainable=True))
        M_gradients = torch.zeros(self.d, self.d)
        B_gradients = torch.zeros(self.d, self.d)
        sigma_gradients = torch.zeros(1)
        # Compute gradients
        for i, retrn in enumerate(returns):
            # Set random seed, get antithetic multiplier and return
            sign = np.sign(seeds[i]).float()
            torch.manual_seed(abs(seeds[i]))
            torch.cuda.manual_seed(abs(seeds[i]))

            
            if self.optimize_sigma == 'per-layer':
                # Max number of elements in a parameter
                m = self.model.max_elements_in_a_tensor()
                l = self.model.count_tensors(only_trainable=True)
                # Get perturbations from search distribution
                eps = self.get_perturbation(torch.Size((l, m)))
                # eps = sign * self.sigma.data * self.B.data.transpose(0, 1) @ eps
                # # Perturb model parameters
                # for l, (pp, s) in enumerate(zip(perturbed_model.parameters(), self.sensitivities)):
                #     # Get needed number of perturbations
                #     e = eps[l, :pp.numel()].view(pp.size())
                #     # print(e.mean(), e.std())
                #     e = self.scale_by_sensitivities(e, s)
                
                weight_gradients += 1 / (self.perturbations * self.sigma) * retrn * eps
            beta_gradients += 1 / (self.perturbations * self.beta.exp()) * retrn * (eps.pow(2).sum() - 1)
        weight_gradients = self._vec2modelrepr(weight_gradients, only_trainable=True)

        # sigma_gradients = sigma_gradients.clamp(max=10)
        # sigma_gradients = self.sigma.data * (sigma_gradients.exp() - 1)

        self.sigma.data = self.sigma.data * np.exp(eta_sigma / 2 * sigma_gradients)

        # # Compute gradients
        # for i, retrn in enumerate(returns):
        #     # Set random seed, get antithetic multiplier and return
        #     sign = np.sign(seeds[i])
        #     torch.manual_seed(abs(seeds[i]))
        #     torch.cuda.manual_seed(abs(seeds[i]))
        #     for layer, param in enumerate(self.model.parameters()):
        #         eps = self.get_perturbation_old(param, sensitivity=self.sensitivities[layer], cuda=self.cuda)
        #         weight_gradients[layer] += self.weight_gradient(sign * retrn, eps)
        #         if self.optimize_sigma:
        #             beta_gradient += self.beta_gradient(retrn, eps)

        # Set gradients
        self.optimizer.zero_grad()
        for layer, param in enumerate(self.model.parameters()):
            param.grad.data = - weight_gradients[layer]
            assert not np.isnan(param.grad.data).any()
            assert not np.isinf(param.grad.data).any()
        # TODO: For each parameter group in the optimizer that is not in the model, update the gradient
        if self.optimize_sigma:
            self.beta.grad = - beta_gradient
            assert not np.isnan(self.beta.grad.data).any()
            assert not np.isinf(self.beta.grad.data).any()

        # # Dependent parameter groups sampling (requires more memory)
        # # Preallocate weight gradients as 1xn vector where n is number of parameters in model
        # IPython.embed()
        # weight_gradients = torch.zeros(self.model.count_parameters(only_trainable=True))
        # beta_gradients = torch.zeros(self.beta.size())
        # # Compute gradients
        # for i, retrn in enumerate(returns):
        #     # Set random seed, get antithetic multiplier and return
        #     sign = np.sign(seeds[i])
        #     torch.manual_seed(abs(seeds[i]))
        #     torch.cuda.manual_seed(abs(seeds[i]))
        #     eps = self.get_perturbation_new(sensitivities=self._modelrepr2vec(self.sensitivities, only_trainable=False), cuda=self.cuda)
        #     weight_gradients += 1 / (self.perturbations * self.sigma) * retrn * eps
        #     beta_gradients += 1 / (self.perturbations * self.beta.exp()) * retrn * (eps.pow(2).sum() - 1)
        # weight_gradients = self._vec2modelrepr(weight_gradients, only_trainable=True)



class GA(ES):
    def __init__(self, model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, **kwargs):
        super(GA, self).__init__(model, env, optimizer, lr_scheduler, eval_fun, perturbations, batch_size, max_generations, safe_mutation, no_antithetic, **kwargs)

    def train(self):
        # Initialize
        # TODO Maybe possible to move to abstract class either in train method or as distinct methods (e.g. init_train)
        self.print_init()
        # Initialize variables dependent on restoring
        if not self._training_start_time:
            # If nothing stored then start from scratch
            start_generation = 0
            max_unperturbed_return = -1e8
            self._training_start_time = time.time()
        else:
            # If something stored then start from last iteration
            start_generation = self.lr_scheduler.last_epoch + 1
            max_unperturbed_return = self._max_unp_return
        # Initialize variables independent of state
        if self.workers > 1: pool = mp.Pool(processes=self.workers)
        if self.track_parallel and "DISPLAY" in os.environ: pb = PoolProgress(pool, update_interval=.5, keep_after_done=False, title='Evaluating perturbations')
        best_algorithm_stdct = None
        best_model_stdct = None
        best_optimizer_stdct = None
        last_checkpoint_time = time.time()
        chunksize = self.perturbations // (10 * self.workers) + int(self.perturbations // (10 * self.workers) == 0)
        eval_kwargs = {'max_episode_length': self.batch_size, 'chunksize': chunksize}
        # eval_kwargs_unp = {'max_episode_length': self.batch_size, 'collect_inputs': hasattr(self.env, 'env'), 'chunksize': chunksize}
        eval_kwargs_unp = {'max_episode_length': self.batch_size, 'collect_inputs': hasattr(self.env, 'env') * self.batch_size, 'chunksize': chunksize}

        if hasattr(self.env, 'env'):
            # unperturbed_out = pool.apply(self.eval_fun, (self.model, self.env, 42), eval_kwargs_unp)
            unperturbed_out = self.eval_fun(self.model, self.env, 42, **eval_kwargs_unp)
            self.sens_inputs = torch.from_numpy(unperturbed_out['inputs'])

        # Start training loop
        for n_generation in range(start_generation, self.max_generations):
            # Compute parent model weight-output sensitivities
            self.compute_sensitivities()

            # Generate random seeds
            seeds = torch.LongTensor(int(self.perturbations/((not self.no_antithetic) + 1))).random_()
            if not self.no_antithetic: seeds = torch.cat([seeds, -seeds])
            assert len(seeds) == self.perturbations, 'Number of created seeds is not equal to wanted perturbations'
            
            workers_start_time = time.time()
            if self.workers > 1:
                # Execute all perturbations on the pool of processes
                workers_out = pool.map_async(partial(self._eval_wrap, **eval_kwargs), seeds)
                unperturbed_out = pool.apply_async(self.eval_fun, (self.model, self.env, 42), eval_kwargs_unp)
                if self.track_parallel and "DISPLAY" in os.environ: pb.track(workers_out)
                workers_out = workers_out.get(timeout=3600)
                unperturbed_out = unperturbed_out.get(timeout=3600)
            else:
                # Execute sequentially
                unperturbed_out = self.eval_fun(self.model, self.env, 42, **eval_kwargs_unp)
                workers_out = []
                for s in seeds:
                    workers_out.append(self._eval_wrap(s, **eval_kwargs))
            workers_time = time.time() - workers_start_time

            assert 'return' in unperturbed_out.keys() and 'seed' in unperturbed_out.keys(), "The `eval_fun` must give a return and repass the used seed"
            if hasattr(self.env, 'env'):
                self.sens_inputs = torch.from_numpy(unperturbed_out['inputs'])
            # Invert output from list of dicts to dict of lists
            workers_out = dict(zip(workers_out[0], zip(*[d.values() for d in workers_out])))
            # Recast all outputs as np.ndarrays except the seeds
            for k, v in filter(lambda i: i[0] != 'seed', workers_out.items()): workers_out[k] = np.array(v)
            
            # Select best model (hillclimber)
            best_idx = np.argmax(workers_out['return'])
            best_seed = workers_out['seed'][best_idx]
            self.model = self.perturb_model(best_seed)
            rank = self.unperturbed_rank(workers_out['return'], unperturbed_out['return'])

            # Keep track of best model
            # TODO bm, bo, ba, mur = self.update_best(unperturbed_out['return'], mur)
            # TODO Maybe not evaluate unperturbed model every iteration
            if unperturbed_out['return'] >= max_unperturbed_return:
                best_model_stdct = self.model.state_dict()
                best_optimizer_stdct = self.optimizer.state_dict()
                best_algorithm_stdct = self.state_dict(exclude=True)
                max_unperturbed_return = unperturbed_out['return']

            # Print and checkpoint
            self._store_stats(workers_out, unperturbed_out, n_generation, rank, workers_time)
            self.print_iter()
            if last_checkpoint_time < time.time() - self.chkpt_int:
                self.save_checkpoint(best_model_stdct, best_optimizer_stdct, best_algorithm_stdct)
                plot_stats(os.path.join(self.chkpt_dir, 'stats.csv'), self.chkpt_dir)
                last_checkpoint_time = time.time()
        
        # End training
        self.save_checkpoint(best_model_stdct, best_optimizer_stdct, best_algorithm_stdct)
        if self.workers > 1:
            pool.close()
            pool.join()
        print("\nTraining done\n\n")



# def generate_seeds_and_models(args, parent_model, self.env):
#     """
#     Returns a seed and 2 perturbed models
#     """
#     np.random.seed()
#     seed = np.random.randint(2**30)
#     two_models = perturb_model(args, parent_model, seed, self.env)
#     return seed, two_models

# if self.optimize_sigma:
#     # print("beta {:5.2f} | bg {:5.1f}".format(args.beta.data.numpy()[0], args.beta.grad.data.numpy()[0]))
#     # print("update_parameters")
#     new_sigma = (0.5*self.beta.exp()).sqrt().data.numpy()[0]
#     # print(" | New sigma {:5.2f}".format(new_sigma), end="")
#     if new_sigma > self.sigma * 1.2:
#         self.sigma = self.sigma * 1.2
#     elif new_sigma < self.sigma * 0.8:
#         self.sigma = self.sigma * 0.8
#     else:
#         self.sigma = new_sigma
#     self.beta.data = torch.Tensor([np.log(2*self.sigma**2)])

            
# # Adjust max length of episodes
# if hasattr(args, 'not_var_ep_len') and not args.not_var_ep_len:
#     args.batch_size = int(5*max(i_observations))


# def perturb_model(self, seed):
#     """Perturbs the main model.

#     Modifies the main model with a pertubation of its parameters,
#     as well as the mirrored perturbation, and returns both perturbed
#     models.
    
#     Args:
#         seed (int): Known random seed. A number between 0 and 2**32.

#     Returns:
#         dict: A dictionary with keys `models` and `seeds` storing exactly that.
#     """
#     # Antithetic or not
#     reps = 1 if self.no_antithetic else 2
#     seeds = [seed] if self.no_antithetic else [seed, -seed]
#     # Get model class and instantiate new models as copies of parent
#     model_class = type(self.model)
#     models = []
#     parameters = [self.model.parameters()]
#     for i in range(reps):
#         this_model = model_class(self.env.observation_space, self.env.action_space) if hasattr(self.env, 'observation_space') else model_class()
#         this_model.load_state_dict(self.model.state_dict())
#         this_model.zero_grad()
#         models.append(this_model)
#         parameters.append(this_model.parameters())
#     parameters = zip(*parameters)
#     # Set seed and permute by isotropic Gaussian noise
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     for pp, p1, *p2 in parameters:
#         eps = self.get_pertubation(pp)
#         p1.data += self.sigma * eps
#         assert not np.isnan(p1.data).any()
#         assert not np.isinf(p1.data).any()
#         if not self.no_antithetic:
#             p2[0].data -= self.sigma * eps
#             assert not np.isnan(p2[0].data).any()
#             assert not np.isinf(p2[0].data).any()
#             assert not (p1.data == p2[0].data).all()
#     return {'models': models, 'seeds': seeds}

def execute_jobs(self, seeds):
    """Start jobs as processes always keeping live processes to a minimum number 
    by continuously ending dead processes
    """
    return_queue = mp.Queue()
    chunksize = self.perturbations // (10 * self.workers) + int(self.perturbations // (10 * self.workers) == 0)
    eval_kwargs = {'max_episode_length': self.batch_size, 'chunksize': chunksize, 'return_queue': return_queue}
    eval_kwargs_unp = {'max_episode_length': self.batch_size, 'collect_inputs': hasattr(self.env, 'env') * self.batch_size, 'chunksize': chunksize, 'return_queue': return_queue}
    processes = []
    outputs = []
    inputs = (self.model, self.env, 42)
    submit_job(self.eval_fun, inputs, eval_kwargs_unp, processes)
    for seed in seeds:
        inputs = (self.perturb_model(seed), self.env, seed)
        submit_job(self.eval_wrap, inputs, eval_kwargs, processes)
        get_output(outputs, processes, return_queue)
        outputs.append(out)

def submit_job(fun, args, kwargs, processes):
    p = mp.Process(target=fun, args=args, kwargs=kwargs)
    p.start()
    processes.append(p)

def get_output(outputs, processes, return_queue):
    # n_done = self.perturbations + 1 - sum([1 for p in processes if p.is_alive()])
    if len(processes) > self.workers * 3:
        pass
    while not return_queue.empty():
        raw_output.append(return_queue.get(False))
    processes = [p for p in processes if p.is_alive()]
    

def start_jobs(self, models, seeds, return_queue):
    processes = []
    while models:
        perturbed_model = models.pop()
        seed = seeds.pop()
        inputs = (perturbed_model, self.env, return_queue, seed)
        try:
            p = mp.Process(target=self.eval_fun, args=inputs, kwargs={'max_episode_length': self.batch_size})
            p.start()
            processes.append(p)
        except (RuntimeError, BlockingIOError) as E:
            IPython.embed()
    assert len(seeds) == 0
    # Evaluate the unperturbed model as well
    inputs = (self.model.cpu(), self.env, return_queue, 'dummy_seed')
    # TODO: Don't collect inputs during run, instead sample at start for sensitivities etc.
    p = mp.Process(target=self.eval_fun, args=inputs, kwargs={'collect_inputs': 1000, 'max_episode_length': self.batch_size})
    p.start()
    processes.append(p)
    return processes, return_queue

def get_job_outputs(self, processes, return_queue):
    raw_output = []
    while processes:
        # Update live processes
        processes = [p for p in processes if p.is_alive()]
        while not return_queue.empty():
            raw_output.append(return_queue.get(False))
    for p in processes:
        p.join()
    return raw_output


# def get_perturbation_old(self, param=None, sensitivity=None, cuda=False):
#     """This method computes a pertubation vector epsilon from a standard normal.

#     It draws perturbations from a standard Gaussian.
#     The pertubation is placed on the GPU if the parameter is there and 
#     `cuda` is `True`. Safe mutations are performed if self.safe_mutation 
#     is not None.
#     """
#     # TODO: This method could also return the perturbations for all model 
#     #       parameters when e.g. sampling from non-isotropic Gaussian. Then, 
#     #       param could default to None. It should still sample form isotropic 
#     #       Gaussian
#     # Sample standard normal distributed pertubation
#     assert (param is None and sensitivity is None) or (param is not None and sensitivity is not None)
#     if param is None:
#         if self.model.is_cuda and cuda:
#             eps = torch.cuda.FloatTensor(self.model.count_parameters(only_trainable=True))
#         else:
#             eps = torch.FloatTensor(self.model.count_parameters(only_trainable=True))
#         eps.normal_(mean=0, std=1)
#         if self.safe_mutation is not None:
#             eps = eps / sensitivity
#             eps = eps / eps.std()
#     else:
#         if param.is_cuda and cuda:
#             eps = torch.cuda.FloatTensor(param.data.size())
#         else:
#             eps = torch.FloatTensor(param.data.size())
#         eps.normal_(mean=0, std=1)
#         # Scale by sensitivities if using safe mutations
#         if self.safe_mutation is not None:
#             assert sensitivity is not None
#             eps = eps / sensitivity       # Scale by sensitivities
#             eps = eps / eps.std()         # Rescale to unit variance
#     return eps


# sES gradients with Wierstra parameterization
# def compute_gradients(self, returns, seeds):
#     """Computes the gradients of the weights of the model wrt. to the return. 
    
#     The gradients will point in the direction of change in the weights resulting in a
#     decrease in the return.
#     """
#     # CUDA
#     if self.cuda:
#         self.model.cuda()

#     # Preallocate list with gradients
#     weight_gradients = []
#     for param in self.model.parameters():
#         weight_gradients.append(torch.zeros(param.data.size()))
#     sigma_gradients = torch.zeros(self.sigma.size())
    
#     # Compute gradients
#     for i, retrn in enumerate(returns):
#         # Set random seed, get antithetic multiplier and return
#         sign = np.sign(seeds[i])
#         torch.manual_seed(abs(seeds[i]))
#         torch.cuda.manual_seed(abs(seeds[i]))
#         i = 0
#         for layer, param in enumerate(self.model.parameters()):
#             eps = self.get_perturbation(param.size(), sensitivities=self.sensitivities[layer], cuda=self.cuda)
#             if not self.optimize_sigma:
#                 weight_gradients[layer] += (1 / (self.perturbations * self.sigma.data)) * (sign * retrn * eps)
#             if self.optimize_sigma == 'single':
#                 weight_gradients[layer] += (1 / (self.perturbations * self.sigma.data)) * (sign * retrn * eps)
#                 sigma_gradients += 1 / ( 2 * self.perturbations) * retrn * (eps.pow(2).sum() - eps.numel())
#             elif self.optimize_sigma == 'per-layer':
#                 weight_gradients[layer] += (1 / (self.perturbations * self.sigma.data[layer])) * (sign * retrn * eps)
#                 # TODO Sigma gradient size is inversely proportional to number of elements in eps
#                 # TODO The more elements in eps, the closer is eps.pow(2).mean() to 1 and the smaller the gradient
#                 # TODO How does eps.pow(2).mean() - 1 scale with elements in eps? Square root?
#                 sigma_gradients[layer] += 1 / ( 2 * self.perturbations) * retrn * (eps.pow(2).sum() - 1)
#             elif self.optimize_sigma == 'per-weight':
#                 j = i + param.numel()
#                 weight_gradients[layer] += (1 / (self.perturbations * self.sigma.data[i:j].view(weight_gradients[layer].size()))) * (sign * retrn * eps)
#                 sigma_gradients[i:j] += 1 / ( 2 * self.perturbations) * retrn * (eps.view(-1).pow(2) - 1)
#                 i = j
#     if self.optimize_sigma:
#         # Clamp gradients before taking exponential transform to avoid inf
#         # IPython.embed()
#         beta = (2 * self.sigma.data.pow(2)).log()
#         beta_gradients = sigma_gradients.clone()
#         beta -= self._sigma['lr'] * beta_gradients
#         self.sigma.data = (0.5 * beta.exp()).sqrt()

#         # sigma_gradients = sigma_gradients.clamp(max=10)
#         # sigma_gradients = self.sigma.data * (sigma_gradients.exp() - 1)
#         # TODO For this parameterization there is a tendency of increasing sigma rather than decreasing
#         # TODO Use regular gradients

#     # Set gradients
#     self.optimizer.zero_grad()
#     for layer, param in enumerate(self.model.parameters()):
#         # param._grad = - Variable(weight_gradients[layer])
#         param.grad.data.set_( - weight_gradients[layer])
#         assert not np.isnan(param.grad.data).any()
#         assert not np.isinf(param.grad.data).any()
#     # if self.optimize_sigma:
#     #     self.sigma.grad = - Variable(sigma_gradients)
#     #     assert not np.isnan(self.sigma.grad.data).any()
#     #     assert not np.isinf(self.sigma.grad.data).any()


#     # ts = time.time()
#     # for i in range(10000):
#     #     for layer, param in enumerate(self.model.parameters()):
#     #         # param._grad = - Variable(weight_gradients[layer])
#     #         param.grad.data.set_( - weight_gradients[layer])
#     #         assert not np.isnan(param.grad.data).any()
#     #         assert not np.isinf(param.grad.data).any()
#     # dur = time.time() - ts
#     # print(dur)
#     # ts = time.time()
#     # for i in range(10000):
#     #     for layer, param in enumerate(self.model.parameters()):
#     #         # param._grad = - Variable(weight_gradients[layer])
#     #         param.grad.data = - weight_gradients[layer]
#     #         assert not np.isnan(param.grad.data).any()
#     #         assert not np.isinf(param.grad.data).any()
#     # dur = time.time() - ts
#     # print(dur)

# Properties for this implementation
        # @property
        # def sigma(self):
        #     if self.optimize_sigma:
        #         assert type(self._sigma) is dict and 'params' in self._sigma
        #         return self._sigma['params'][0]
        #     else:
        #         assert type(self._sigma) is not dict
        #         return self._sigma

        # @sigma.setter
        # def sigma(self, sigma):
        #     if self.optimize_sigma:
        #         assert isinstance(sigma, (collections.Sequence, np.ndarray, torch.Tensor))
        #         if isinstance(sigma, (collections.Sequence, np.ndarray)):
        #             self._sigma['params'][0].data = torch.Tensor(sigma)
        #         else:
        #             self._sigma['params'][0].data = sigma
        #     else:
        #         assert not isinstance(sigma, (collections.Sequence, np.ndarray, torch.Tensor))
        #         self._sigma = sigma
