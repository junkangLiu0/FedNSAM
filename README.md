

+vx 15653218567 帮你解决代码部署问题！
# Setup

- torch == 2.1
- torchvision == 0.19
- ray == 1.0.0

# Dataset

- CIFAR10
- CIFAR100
- Tiny ImageNet

# Model

- LeNet-5, VGG-11, and ResNet-18
- Vision Transformer (ViT-Base), Swin transformer (Swin-Small, Swin-Base)

# Run

```python
python new_NSAM.py --alg FedNSAM --lr 0.1 --data_name CIFAR100 --alpha_value 0.1 --alpha 0.9 --epoch 1001  --extname CIFAR100 --lr_decay 0.998 --gamma 0.85 --CNN resnet10 --E 5 --batch_size 50 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --rho 0.01
```

Explanations of arguments:

- `alg`: FedAvg, FedAvgM, SCAFFOLD, FedACG, FedSAM, MoFedSAM, FedGAMMA, FedLESAM, FedNSAM

- `alpha_value`: parameter of Dirichlet Distribution, controling the level of Non-IID

- `E`: local training epochs for each client

- `selection`: the selection fraction of total clients in each round
