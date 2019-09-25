# Benchmarking Adversarial Robustness (submitted to ICLR2020)

This repository contains the code for reproducing the results of evaluating all algorithms under most of settings, of our *BENCHMARKING ADVERSARIAL ROBUSTNESS*.

## Installation
* Python 3.5
* TensorFlow >= 1.3.0 (with GPU support)
* PyTorch 1.0.1
* tensorpack

## Introduction
`realsafe` is a package that includes all attack and defense algorithms that are incorporated into our benchmark for evaluation. 

`benchmark-v1` and `benchmark-v2` include all the benchmark codes against all defense algorithms mentioned in the paper. Specifically, 
`benchmark_v1` contains benchmark scripts by using DeepFool and all black-box attack algorithms we use, and `benchmark_v2` is based on white-box algorithms. 

Among them, 

- **[⬆ Effectiveness]**:

`_distoration` scripts are to get the accuracy vs. perturbation budget, we perform a line search followed by a binary search on pertubation to find its minimum value. 

- **[⬆ Efficiency]**: 

`_iteration` scripts can obtain the accuracy vs. attack strength, where the attacks trength is defined as the number of iterations or model queries for different attack methods

Note that these two folders do not use the same interface, this will be fixed later.

## Reproducing the results on Cifar dataset

### benchmark-v1 example

```
cd benchmark-v1
python3 cifar_resnet56_iteration.py --method nes --goal ut --distance l_2  --xs PATH_CIFAR_XS --ys PATH_CIFAR_YS --ys-target PATH_CIFAR_YS_TARGETS --output CIFAR_OUTPUT.npy

```

### benchmark-v2 example
```  
cd benchmark-v2
python3 cifar_convex_distortion.py --method bim --goal t --distance-metric l_inf --xs PATH_CIFAR_XS --ys PATH_CIFAR_YS --ys-target PATH_CIFAR_YS_TARGETS -- output CIFAR_OUTPUT.npy
```

We provide 100 data points of CIFAR-10 in `datasets/CIFAR-10/`. You can set `PATH_CIFAR_XS` as `datasets/CIFAR-10/00000_00100_xs.npy`, `PATH_CIFAR_YS` as `datasets/CIFAR-10/00000_00100_ys.npy`, `PATH_CIFAR_YS_TARGETD` as `datasets/CIFAR-10/00000_00100_ys_target.npy`

## Reproducing the results on ImageNet dataset

### benchmark-v1 example

```
cd benchmark-v1
python3 imagenet_incv3_iteration.py --method boundary --goal ut --distance l_2 --start 0 --end 100 --output IMAGENET_OUT.npy 

```

### benchmark-v2 example
```
cd benchmark-v2
python3 imagenet_incv3_distortion.py --method mim --goal t --distance l_inf --output IMAGENET_OUT.npy

```
