# Towards Quantifying the Hessian Structure of Neural Networks
This repository contains PyTorch implementations of Hessian visualization of neural networks. 

## How to use 

Install torch (>=1.8.0) and run the following command for the results on CIFAR-100.

```
python train_cifar.py
```

`train_cifar.py` now computes the Hessian at selected epochs during training. By
default it evaluates the Hessian halfway through training and at the final epoch,
which corresponds to the learning‑rate decay phase. The epochs can be adjusted via
the `hessian_epochs` list in the script.

Run the following command for the results on Gaussian synthetic data.

```
python train_gaussian.py
```


