# L2T: An RNN-Based Framework for the MILP Problem in Robustness Verification of Neural Networks
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
To reproduce the experiments on OVAL benchmark and COLT benchmark, please run the following command:
```bash
python run_batched_verification.py
```
