# L2T: An RNN-Based Framework for the MILP Problem in Robustness Verification of Neural Networks

> **Abstract**: *Robustness verification of deep neural networks is becoming increasingly crucial for their potential use in many safety-critical applications. Essentially, the problem of robustness verification can be encoded as a typical Mixed-Integer Linear Programming (MILP) problem, which can be solved via branch-and-bound strategies. However, these methods can only afford limited scalability and remain challenging for verifying large-scale neural networks. In this paper, we present a novel framework to speed up the solving of the MILP problems generated from the robustness verification of deep neural networks. It employs a semi-planet relaxation to abstract ReLU activation functions, via an RNN-based strategy for selecting the relaxed ReLU neurons to be tightened. We have developed a prototype tool L2T and conducted comparison experiments with state-of-the-art verifiers on a set of large-scale benchmarks. The experiments show that our framework is both efficient and scalable even when applied to verify the robustness of large-scale neural networks.*

## Installation

```bash
cd L2T
conda create -n L2T
conda activate L2T  
pip install -r requirements.txt
```

## Running the Benchmarks

To reproduce the experiments on OVAL benchmark and COLT benchmark, please run the following command:

```bash
python run_verification.py
```

## Datasets

Our training datasets is available at [https://drive.google.com/file/d/1LtOtt6GFt2M68I3avScf-U8Ydn0QDsET/view?usp=sharing](https://drive.google.com/file/d/1LtOtt6GFt2M68I3avScf-U8Ydn0QDsET/view?usp=sharing)
