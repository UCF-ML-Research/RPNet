# RPNet
This is the implementation of our paper "Audit and Improve Robustness of Private Neural Networks on Encrypted Data" [Arxiv](https://arxiv.org/abs/2209.09996).

The code is organized and maintained by Jiaqi Xue of Dr. Qian Lou's Lab at the University of Central Florida (UCF).

## Overview
- We first identified that existing attacks and defense techniques for Neural Network are not transformed well to Private Neural Networks (PNet). 
- We propose PNet-Attack to efficiently attack PNet in both target and untarget manners by arc-shaped search in the frequency domain and a cosine annealing perturbation size schedule. 
- To defend the adversarial attacks, we propose RPNet by adding noise in the output layer and a DNT technique to design a Robust and Private Neural Network.
![overview](https://github.com/UCF-ML-Research/RPNet/blob/main/figure/RPNet.png)
## Environment Setup
Our codebase requires the following Python and PyTorch versions:
- Python --> 3.11.3
- PyTorch --> 2.0.1

## Model Training
### MNIST

- BaseModel: a simple CNN used on MNIST ACC: 98.32%
- eBaseModel: replace all relu() with square()   ACC: 98.21%

To train model and save:

```bash
python models/train.py --dataset MNIST --epochs 20 --lr 0.001 --model BaseNet --save_dir ./checkpoint/MNIST/BaseNet.pth
```

Then, we can compress both model with 8bits:

```bash
python ./compressed/compress_model.py --dataset MNIST --model BaseNet --model_dir ./checkpoint/MNIST/BaseNet.pth --dataset_dir ./data --save_dir ./checkpoint/MNIST/BaseNet-8.pth --act_bits 8 --weight_bits 8
```
- BaseModel-8:   ACC: 97.59%
- eBaseModel-8:  ACC: 98.11%

### CIFAR10
- BaseModel: a simple CNN used on CIFAR10 ACC: 81.15%
- eBaseModel: replace all relu() with square()   ACC: 77.27%

To train model and save:

```bash
python models/train.py --dataset CIFAR10 --epochs 100 --lr 0.001 --model eBaseNet --save_dir ./checkpoint/CIFAR10/eBaseNet.pth
```

After training model, we can run this to compress the model

```bash
python ./compressed/compress_model.py --dataset CIFAR10 --model BaseNet --model_dir ./checkpoint/CIFAR10/BaseNet.pth --dataset_dir ./data --save_dir ./checkpoint/CIFAR10/BaseNet-8.pth --act_bits 8 --weight_bits 8
```

This code operation will compress the BaseNet into 8bits.


## SimBa Attack
Attack our models: eBaseNet, BaseNet, eBaseNet-10bit and BaseNet-8bit with [SimBa attack](https://arxiv.org/pdf/1905.07121).

1. BaseNet

```bash
python attack/run_simba_cifar.py --targeted --model BaseNet --model_ckpt ./checkpoint/CIFAR10/BaseNet.pth --epsilon 0.2 
```

2. eBaseNet

```bash
python attack/run_simba_cifar.py --targeted --model eBaseNet --model_ckpt ./checkpoint/CIFAR10/eBaseNet.pth --epsilon 0.2
```

3. BaseNet-8

```bash
python attack/run_simba_cifar.py --targeted --compress --model BaseNet --model_ckpt ./checkpoint/BaseNet-8.pth --epsilon 0.7
```

4. eBaseNet-10

```bash
python attack/run_simba_cifar.py --targeted --compress --model eBaseNet --model_ckpt ./checkpoint/eBaseNet-10.pth --epsilon 0.7
```

## PNet Attack and RPNet Defense

### Key parameters
`--num_runs`: number of image samples

`--batch_size`: batch size for parallel runs

`--data_root`: the location of your dataset

`--targeted`: targeted attack or untargeted attack

`--model_ckpt`: pth files.

`--sigma1`: gaussian noise adding on input layer

`--sigma2`: gaussian noise adding on confidence layer

`--T`: the cycle of epsilon schedule

`--beta_min`, beta_max: beta_range[beta_min, beta_max]

`--image_size`: 28 for mnist and 32 for CIFAR10

```bash
python ./attack/simba_dev.py --num_runs 128 --batch_size 128 --data_root ./data --dataset cifar --model_type CIFAR10 --image_size 32 --targeted --model_ckpt ./checkpoint/CIFAR10/RND/eBaseNet-10.pth --sigma1 0.1 --sigma2 0.05 --T 400 -beta_min 0.5 -beta_max 1.5 --epsilon 1
```

## Citation
If you find this code useful, please consider citing our paper:
```
@article{xue2022audit,
  title={Audit and improve robustness of private neural networks on encrypted data},
  author={Xue, Jiaqi and Xu, Lei and Chen, Lin and Shi, Weidong and Xu, Kaidi and Lou, Qian},
  journal={arXiv preprint arXiv:2209.09996},
  year={2022}
}
```
