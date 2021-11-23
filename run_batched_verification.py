import os, sys

"""
Run the bab_runner experiments
"""


def run_bab_exp(gpu_id, cpus, timeout, pdprops, nn, method, data, branching=None):

    pdprops_list = [
        "jodie-base_easy.pkl",
        "jodie-base_med.pkl",
        "jodie-base_hard.pkl",
        "jodie-wide.pkl",
        "jodie-deep.pkl",
        "base_100.pkl",
        "wide_100.pkl",
        "deep_100.pkl",
    ]
    nn_names = [
        "cifar_base_kw",
        "cifar_wide_kw",
        "cifar_deep_kw",
        "mnist_0.1",
        "mnist_0.3",
        "cifar10_2_255",
        "cifar10_8_255",
        ]
    methods = [
        "prox",     # Proximal == BDD+
        "adam",     # 
        "dj-adam",
        "gurobi",
        "gurobi-anderson",  # 
    ]
    if pdprops:
        assert pdprops in pdprops_list
    assert nn in nn_names
    assert method in methods
    pdprops_str = f"--pdprops {pdprops}" if pdprops else ""

    if method in ["gurobi", "gurobi-anderson"]:
        batch_size = 150
        parent_init = ""
        alg_specs = "--gurobi_p 6"  # Gurobi runs on 6 cores
        if method == "gurobi-anderson":
            alg_specs += f" --n_cuts 1"
    else:
        parent_init = "--parent_init"
        if nn in ["cifar_base_kw", "mnist_0.1", "mnist_0.3"]:
            batch_size = 150
        elif nn in ['cifar10_2_255', 'cifar10_8_255']:
            batch_size = 32
        elif nn in ['cifar_wide_kw']:
            batch_size = 64
        else:
            batch_size = 100
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
        max_solver_batch = 12000
    elif nn in ["mnist_0.1"]:
        max_solver_batch = 8000
    elif nn in ["mnist_0.3"]:
        max_solver_batch = 6000
    elif nn in ['cifar_deep_kw']:
        # the plots were run with 18000 but let's stay on the safe side.
        max_solver_batch = 17000
    elif nn in ['cifar10_2_255']:
        max_solver_batch = 1800
    elif nn in ['cifar10_8_255']:
        max_solver_batch = 3600

    command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python bab_runner.py " \
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
    cpus = "6-9"

    ## mnist-eth
    data = "mnist"
    method = "prox"
    nns = ["mnist_0.1", "mnist_0.3"]
    pdprops = None
    timeout = 300
    branching = 'nn'
    # branching = 'heuristic'
    for nn in nns:
        run_bab_exp(gpu_id, cpus, timeout, pdprops, nn, method, data, branching=branching)

    ## cifar-eth
    data = "cifar10"
    method = "prox"
    nns = ["cifar10_2_255", "cifar10_8_255"]
    pdprops = None
    timeout = 300
    branching = 'nn'
    # branching = 'heuristic'
    for nn in nns:
        run_bab_exp(gpu_id, cpus, timeout, pdprops, nn, method, data, branching=branching)

    # cifar
    data = "cifar"
    timeout = 3600
    method = "prox"
    specs = [
        ("cifar_base_kw", "base_100.pkl"),
        ("cifar_wide_kw", "wide_100.pkl"),
        ("cifar_deep_kw", "deep_100.pkl"),
    ]
    branching = 'nn'
    # branching = 'heuristic'
    for nn, pdprops in specs:
        run_bab_exp(gpu_id, cpus, timeout, pdprops, nn, method, data, branching=branching)


if __name__ == "__main__":

    run_vnn_results()
