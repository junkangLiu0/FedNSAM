

FedNSAM！

# Setup

* 有代码问题+vx15653218567 马上回复！帮忙引用论文一下就行！

* 一张4090或者两张2080ti即可训练！！发顶会！！代码问题或者讨论+vx 15653218567

* 我的其他论文也都是这一套代码配置，均可复现！差分隐私，联邦泛化，联邦大模型，联邦优化，联邦大模型微调lora。。。。

* 个人主页：https://junkangliu0.github.io/
* 
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
python  main_FedNSAM.py --alg FedACG --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedNSAM --lr_decay 2 --gamma 0.85  --CNN   resnet18 --E 5 --batch_size 50   --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10  --rho 0.01 --pix 32 --lora 0 --K 50
python  main_FedNSAM.py --alg FedNesterov --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedNSAM --lr_decay 2 --gamma 0.85  --CNN   resnet18 --E 5 --batch_size 50   --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10  --rho 0.01 --pix 32 --lora 0 --K 50
python  main_FedNSAM.py --alg FedNSAM --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedNSAM --lr_decay 2 --gamma 0.85  --CNN   resnet18 --E 5 --batch_size 50   --gpu 2 --p 0 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10  --rho 0.01 --pix 32 --lora 0 --K 50

```

Explanations of arguments:

- `alg`: FedAvg, FedAvgM, SCAFFOLD, FedACG, FedSAM, MoFedSAM, FedGAMMA, FedLESAM, FedNSAM

- `alpha_value`: parameter of Dirichlet Distribution, controling the level of Non-IID

- `E`: local training epochs for each client

- `selection`: the selection fraction of total clients in each round

- ## Parameter Reference

### Core Federated Learning Parameters
| Parameter | Description |
|-----------|-------------|
| `--alg` | Algorithm choice: FedAvg, FedAvgM, SCAFFOLD, FedACG, FedSAM, MoFedSAM, FedGAMMA, FedLESAM, FedNSAM, etc. |
| `--lr` | Client learning rate |
| `--lr_decay` | Learning rate decay strategy (1=exponential, 2=cosine annealing) |
| `--gamma` | Momentum parameter for certain algorithms |
| `--alpha` | Weight decay coefficient for AdamW optimizer |

### Data Parameters
| Parameter | Description |
|-----------|-------------|
| `--data_name` | Dataset: `CIFAR10`, `CIFAR100`, `imagenet`, `QQP`, `MNLI`, etc. |
| `--alpha_value` | Dirichlet distribution parameter for non-IID data splitting (0.1=highly non-IID, 1=IID) |
| `--num_workers` | Total number of clients |
| `--selection` | Fraction of clients selected per round (0.1=10%) |

### Model Parameters
| Parameter | Description |
|-----------|-------------|
| `--CNN` | Model architecture: `resnet18`, `swin_tiny`, `deit_tiny`, `roberta_base` |
| `--pre` | Use pretrained weights (1=True, 0=False) |
| `--normalization` | Normalization type: `BN` (BatchNorm) or `GN` (GroupNorm) |
| `--pix` | Input image size (32 for CIFAR, 224 for ImageNet) |

### Training Parameters
| Parameter | Description |
|-----------|-------------|
| `--epoch` | Total communication rounds |
| `--E` | Local epochs per client |
| `--batch_size` | Client batch size |
| `--K` | Maximum local steps per round (overrides E if smaller) |
| `--p` | Parallelism factor for client updates |

### LoRA Parameters
| Parameter | Description |
|-----------|-------------|
| `--lora` | Enable LoRA fine-tuning (1=True, 0=False) |
| `--r` | LoRA rank |
| `--lora_alpha` | LoRA scaling parameter |

### Optimization Parameters
| Parameter | Description |
|-----------|-------------|

| `--rho` | SAM optimizer perturbation radius |
| `--optimizer` | Base optimizer: `SGD` or `AdamW` |

### System Parameters
| Parameter | Description |
|-----------|-------------|
| `--gpu` | GPU device IDs (e.g., "0,1,2") |
| `--num_gpus_per` | GPU fraction per client (0.2=20% of a GPU) |
| `--print` | Print detailed logs (1=True, 0=False) |
| `--preprint` | Evaluation frequency (in epochs) |

---

## Output Files

- **Logs**: `./log/alg-dataset-lr-workers-batch-epochs-lr_decay.txt`
- **Checkpoints**: `./checkpoint/ckpt-alg-lr-extname-alpha_value-timestamp/`
- **Plots**: `./plot/alg-dataset-...-timestamp.npy` (contains accuracy/loss arrays)
- **Models**: `./model/model-alg-...-timestamp.pth`
