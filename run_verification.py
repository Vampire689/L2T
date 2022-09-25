import os, sys

"""
Run the L2T_runner experiments
"""


def run_L2T_exp(gpu_id, cpus, timeout, pdprops, nn, method, data, branching=None):

    pdprops_list = [
        "base_100.pkl",
        "wide_100.pkl",
        "deep_100.pkl",
    ]
    nn_names = [
        "cifar_base_kw",
        "cifar_wide_kw",
        "cifar_deep_kw",
        "cifar10_2_255",
        "cifar10_8_255",
        ]
    methods = [
        "prox",    
        "adam",      
        "dj-adam",
        "gurobi",
        "gurobi-anderson",  
    ]
    if pdprops:
        assert pdprops in pdprops_list
    assert nn in nn_names
    assert method in methods
    pdprops_str = f"--pdprops {pdprops}" if pdprops else ""

    if method in ["gurobi", "gurobi-anderson"]:
        batch_size = 150
        parent_init = ""
        alg_specs = "--gurobi_p 6"
        if method == "gurobi-anderson":
            alg_specs += f" --n_cuts 1"
    else:
        parent_init = "--parent_init"
        if nn in ["cifar_base_kw", "cifar_wide_kw", "cifar_deep_kw"]:
            batch_size = 256
        elif nn in ['cifar10_2_255', 'cifar10_8_255']:
            batch_size = 32
        if method == "prox":
            alg_specs = "--tot_iter 100"
            if "cifar10" in nn:
                alg_specs += " --eta 1e1 --feta 1e1"
            else:
                alg_specs += " --eta 1e2 --feta 1e2"
        elif method == "adam":
            adam_iters = 175 if "mnist" in nn else 160
            alg_specs = f"--tot_iter {adam_iters}"
            if "mnist" in nn:
                alg_specs += " --init_step 1e-1 --fin_step 1e-3"
            elif "cifar10" in nn:
                alg_specs += " --init_step 1e-4 --fin_step 1e-6"
            else:
                alg_specs += " --init_step 1e-3 --fin_step 1e-4"
        elif method == "dj-adam":
            alg_specs = "--tot_iter 260"
            if "mnist" in nn:
                alg_specs += " --init_step 1e-1 --fin_step 1e-3"
            else:
                alg_specs += " --init_step 1e-3 --fin_step 1e-4"

    if nn in ["cifar_base_kw"]:
        max_solver_batch = 25000
    elif nn in ["cifar_wide_kw"]:
        max_solver_batch = 15000
    elif nn in ['cifar_deep_kw']:
        max_solver_batch = 15000
    elif nn in ['cifar10_2_255']:
        max_solver_batch = 1800
    elif nn in ['cifar10_8_255']:
        max_solver_batch = 3600

    command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python L2T_runner.py " \
              f"--timeout {timeout} {pdprops_str} --nn_name {nn} --record --method {method} {alg_specs} " \
              f"--batch_size {batch_size} --max_solver_batch {max_solver_batch} {parent_init} --data {data}"
    if branching and branching == 'nn':
        command += ' --branching_choice nn'
    else:
        command += ' --branching_choice heuristic'
    print(command)
    os.system(command)



def run_vnn_results():

    # Edit these parameters according to the available hardware.
    gpu_id = 0
    cpus = "0-9"

    ## OVAL
    data = "cifar"
    timeout = 3600
    method = "prox"
    specs = [
        ("cifar_base_kw", "base_100.pkl"),
        ("cifar_wide_kw", "wide_100.pkl"),
        ("cifar_deep_kw", "deep_100.pkl"),
    ]
    branching = 'nn'
    for nn, pdprops in specs:
        run_L2T_exp(gpu_id, cpus, timeout, pdprops, nn, method, data, branching=branching)

    ## COLT
    data = "cifar10"
    method = "prox"
    nns = ["cifar10_2_255", "cifar10_8_255"]
    pdprops = None
    timeout = 300
    branching = 'nn'
    for nn in nns:
        run_L2T_exp(gpu_id, cpus, timeout, pdprops, nn, method, data, branching=branching)

    


if __name__ == "__main__":
    run_vnn_results()
