# L2T: An RNN-Based Framework for the MILP Problem in Robustness Verification of Neural Networks
> **Abstract**: *Robustness verification of deep neural networks is becoming increasingly crucial for their potential use in many safety-critical applications. Essentially, the problem of robustness verification can be encoded as a typical Mixed-Integer Linear Programming (MILP) problem, which can be solved via branch-and-bound strategies. However, these methods can afford limited scalability and remain challenging for large-scale neural networks. In this paper, we present a novel framework to speed up the solving of the MILP problems generated from the robustness verification of deep neural networks. It employs a semi-planet relaxation to abstract ReLU activation functions, via an RNN-based strategy for selecting the relaxed ReLU neurons to be tightened. We have implemented a prototype tool L2T and conducted comparison experiments with some state-of-the-art verifiers on a set of large-scale benchmarks, showing our advantages on efficiency and scalability of robustness verification on large-scale neural networks with tens of thousands of neurons.*

## Installation

```bash
git clone https://github.com/Vampire689/L2T.git
cd L2T  

conda create -n L2T
conda activate L2T  

conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch
python setup.py install
```

## Running the Benchmarks
The experimental settings are replicated from https://github.com/verivital/vnn-comp/tree/master/2020/CNN/oval_framework. 
To reproduce the experiments on OVAL benchmark and COLT benchmark, please run the following command:
```bash
python run_batched_verification.py
```

## Results
Results are saved in xlsx as well in the created folder ./results.
