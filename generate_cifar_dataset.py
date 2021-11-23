import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import argparse
import torch
from torch import nn, optim
from plnn.branch_and_bound.relu_train_generation import relu_traingen
from plnn.branch_and_bound.relu_branch_and_bound import generate_relu_bab
from tools.bab_tools.model_utils import load_cifar_1to1_exp, load_1to1_eth
from plnn.proxlp_solver.solver import SaddleLP
from plnn.proxlp_solver.dj_relaxation import DJRelaxationLP
from plnn.network_linear_approximation import LinearizedNetwork
from plnn.anderson_linear_approximation import AndersonLinearizedNetwork
from nn_branching.exp_conv import ExpNet
import time
import pandas as pd
import copy
import math
import torch.multiprocessing as mp
import itertools
import csv
'''
Code from NN_branching (author: Jodie)
This script supports following verifications methods.

Branch and Bound with a heuristic splitting strategy, developed based on Kolter and Wong's paper (--bab_kw)
'''
class MyLossFunc(nn.Module):
    def __init__(self):
        super(MyLossFunc, self).__init__()
    def forward(self, logit, mv):
        # new loss
        k = 50
        l1_loss = torch.abs(logit - mv)
        _, indices = mv.topk(k)
        rank_w = (mv.clone().detach() + 0.5)*0.5
        rank_w[indices] = rank_w[indices]*2
        for i, idx in enumerate(indices):
            # rank_w[idx] = rank_w[idx] * (2*(k - i)/k + 1)
            rank_w[idx] = rank_w[idx] * ((((k - i)/k)**3)*4 + 1)
        rank_loss = ((logit < mv)*1 + (logit >= mv)*(l1_loss > 0.1)*1) * rank_w * l1_loss
        
        gap_loss, num= 0, 0
        # _, indices = mv.topk(2*k)
        for i, idx in enumerate(indices):
            for j, jdx in enumerate(indices):
                if i<j and abs(logit[idx]-logit[jdx]) < 0.05:
                # if i<j and mv[idx] - mv[jdx] > 0.1:
                    gap_loss += max(0, 0.1-(logit[idx]-logit[jdx]))
                    num += 1
        gap_loss = gap_loss / max(1, num)
        return torch.mean(rank_loss) + gap_loss

class BranchNet:
    def __init__(self, model_path):
        model = ExpNet(128)
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            pass
        self.model = model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, eps=1e-4, weight_decay=1e-4)
        self.criterion = MyLossFunc()


# Pre-fixed parameters
pref_branching_thd = 0.2
pref_online_thd = 2
pref_kwbd_thd = 20
models = {}
models['cifar_base_expnn'] = './nn_branching/nn_models/cifar/cifar_trained_expnn.pt'
branch_net = BranchNet(model_path = models['cifar_base_expnn'])


def bab(gt_prop, verif_layers, domain, return_dict, timeout, batch_size, method, tot_iter,  parent_init,
        args, gurobi_dict=None, imag_idx = None):
    epsilon = 1e-4
    decision_bound = 0
    gpu = True

    if gpu:
        cuda_verif_layers = [copy.deepcopy(lay).cuda() for lay in verif_layers]
        domain = domain.cuda()
    else:
        cuda_verif_layers = [copy.deepcopy(lay) for lay in verif_layers]

    # use best of naive interval propagation and KW as intermediate bounds
    intermediate_net = SaddleLP(cuda_verif_layers, store_bounds_primal=True, max_batch=args.max_solver_batch)
    intermediate_net.set_solution_optimizer('best_naive_kw', None)

    if method == "prox":
        bounds_net = SaddleLP(cuda_verif_layers, store_bounds_primal=True, max_batch=args.max_solver_batch)
        bounds_net.set_decomposition('pairs', 'KW')
        optprox_params = {
            'nb_total_steps': int(tot_iter),
            'max_nb_inner_steps': 2,  # this is 2/5 as simpleprox
            'initial_eta': args.eta,
            'final_eta': args.feta,
            'log_values': False,
            'outer_cutoff': args.cutoff,
            'maintain_primal': True
        }
        bounds_net.set_solution_optimizer('optimized_prox', optprox_params)
        print(f"Running prox with {tot_iter} steps")

    # branching
    if args.branching_choice == 'heuristic':
        branching_net_name = None
    else:
        if method == 'prox':
            branching_net_name = models['cifar_base_expnn']
        else: 
            raise NotImplementedError

    min_lb, min_ub, ub_point, nb_states, fail_safe_ratio = generate_relu_bab(intermediate_net, bounds_net, branching_net_name, imag_idx, domain, decision_bound,
                                                    eps=epsilon, timeout=timeout,
                                                    batch_size=batch_size,
                                                    parent_init_flag=parent_init, gurobi_specs=gurobi_dict, branch_net=branch_net)

    if not (min_lb or min_ub or ub_point):
        return_dict["min_lb"] = None;
        return_dict["min_ub"] = None;
        return_dict["ub_point"] = None;
        return_dict["nb_states"] = nb_states
        return_dict["bab_out"] = "timeout"
        return_dict["fs_ratio"] = fail_safe_ratio
    else:
        return_dict["min_lb"] = min_lb.cpu()
        return_dict["min_ub"] = min_ub.cpu()
        return_dict["ub_point"] = ub_point.cpu()
        return_dict["nb_states"] = nb_states
        return_dict["fs_ratio"] = fail_safe_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true', help='file to save results', default=False)
    parser.add_argument('--train_generation', action='store_true', help='mode of generating training datasets', default=False)
    parser.add_argument('--train_gt_throughout', action='store_true', help='generate groud truths for each branch throughout a property verification')
    parser.add_argument('--record_name', type=str, help='file to save results')
    # parser.add_argument('--pdprops', type=str, help='pandas table with all props we are interested in', default="base_100.pkl")
    parser.add_argument('--pdprops', type=str, help='pandas table with all props we are interested in', default="train_base_easy.pkl")
    parser.add_argument('--timeout', type=int, default=7200)
    parser.add_argument('--cpus_total', type=int, help='total number of cpus used')
    parser.add_argument('--cpu_id', type=int, help='the index of the cpu from 0 to cpus_total')
    parser.add_argument('--nn_name', type=str, help='network architecture name', default="cifar_base_kw")
    parser.add_argument('--data', type=str, default='cifar')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--batch_size', type=int, help='batch size / 2 for how many domain computations in parallel', default=64)
    parser.add_argument('--gurobi_p', type=int, help='number of threads to use in parallelizing gurobi over domains', default=1)
    parser.add_argument('--method', type=str, choices=["prox", "adam", "gurobi", "gurobi-anderson", "dj-adam"], help='method to employ for bounds', default="prox")
    parser.add_argument('--branching_choice', type=str, default='nn', help='type of branching choice used')
    parser.add_argument('--tot_iter', type=float, help='how many total iters to use for the method', default=100)
    parser.add_argument('--max_solver_batch', type=float, default=25000, help='max batch size for bounding computations')
    parser.add_argument('--parent_init', action='store_true', help='whether to initialize the code from the parent', default=True)
    parser.add_argument('--n_cuts', type=int, help='number of anderson cuts to employ (per neuron)')
    parser.add_argument('--eta', type=float, default=1e2)
    parser.add_argument('--feta', type=float, default=1e2)
    parser.add_argument('--cutoff', type=float, default=0)
    parser.add_argument('--init_step', type=float)
    parser.add_argument('--fin_step', type=float)
    args = parser.parse_args()

    # initialize a file to record all results, record should be a pandas dataframe
    if args.data == 'cifar' or args.data=='cifar10':
        path = './cifardata/'
        result_path = './train_results/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    else:
        raise NotImplementedError

    # load all properties
    if args.data=='cifar':
        gt_results = pd.read_pickle(path + args.pdprops)
        bnb_ids = gt_results.index
        batch_ids = bnb_ids

    gurobi_dict = {"gurobi": args.method in ["gurobi", "gurobi-anderson"], "p": args.gurobi_p}

    for new_idx, idx in enumerate(batch_ids[270:300]): 
        if args.data == 'cifar':
            imag_idx = gt_results.loc[idx]["Idx"]
            prop_idx = gt_results.loc[idx]['prop']
            eps_temp = gt_results.loc[idx]["Eps"]

            # skip the nan prop_idx or eps_temp (happens in wide.pkl, jodie's mistake, I guess)
            if (math.isnan(imag_idx) or math.isnan(prop_idx) or math.isnan(eps_temp)):
                continue

            x, verif_layers, test = load_cifar_1to1_exp(args.nn_name, int(imag_idx), int(prop_idx))
            # since we normalise cifar data set, it is unbounded now
            bounded = False
            assert test == prop_idx
            domain = torch.stack([x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp], dim=-1)
            linear = False
        else:
            raise NotImplementedError

        ### BaB
        gt_prop = f'idx_{imag_idx}_prop_{prop_idx}_eps_{eps_temp}'
        print(gt_prop)
        bab_start = time.time()
        return_dict = dict()
        bab(gt_prop, verif_layers, domain, return_dict, args.timeout, args.batch_size, args.method, args.tot_iter, args.parent_init, args, gurobi_dict=gurobi_dict, imag_idx=imag_idx)


if __name__ == '__main__':
    main()
