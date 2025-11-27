

+vx 15653218567 帮你解决代码部署问题！
# Setup

## Requirements

* Python 3.8
* PyTorch
* torchvision
* numpy
* matplotlib
* tensorboardX
* ray==1.0.0
* filelock

You can install the dependencies with:

```bash
pip install -r requirements.txt
```

# Dataset

- CIFAR10
- CIFAR100
- Tiny ImageNet

# Model

- LeNet-5, VGG-11, and ResNet-18
- Vision Transformer (ViT-Base), Swin transformer (Swin-Small, Swin-Base)

# Run

```python
python  main_FedNSAM.py --alg FedACG --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedMuon --lr_decay 2 --gamma 0.85  --CNN   resnet18 --E 5 --batch_size 50   --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10  --rho 0.01 --pix 32 --lora 0 --K 50
python  main_FedNSAM.py --alg FedNesterov --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedMuon --lr_decay 2 --gamma 0.85  --CNN   resnet18 --E 5 --batch_size 50   --gpu 1 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10  --rho 0.01 --pix 32 --lora 0 --K 50
python  main_FedNSAM.py --alg FedNSAM --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedMuon --lr_decay 2 --gamma 0.85  --CNN   resnet18 --E 5 --batch_size 50   --gpu 2 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10  --rho 0.01 --pix 32 --lora 0 --K 50

```

Explanations of arguments:

- `alg`: FedAvg, FedAvgM, SCAFFOLD, FedACG, FedSAM, MoFedSAM, FedGAMMA, FedLESAM, FedNSAM

- `alpha_value`: parameter of Dirichlet Distribution, controling the level of Non-IID

- `E`: local training epochs for each client

- `selection`: the selection fraction of total clients in each round
