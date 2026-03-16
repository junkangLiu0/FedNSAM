

FedNSAM！

# Setup

* 有代码问题+vx15653218567 马上回复！帮忙引用论文一下就行！

* 一张4090或者两张2080ti即可训练！！发顶会！！代码问题或者讨论+vx 15653218567

* 我的其他论文也都是这一套代码配置，均可复现！差分隐私，联邦泛化，联邦大模型，联邦优化，联邦大模型微调lora。。。。

* 个人主页：https://junkangliu0.github.io/
  
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
python  main_FedNSAM.py --alg FedACG --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedNSAM --lr_decay 2 --gamma 0.85  --CNN   resnet18 --E 5 --batch_size 50   --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10  --rho 0.01 --lora 0 --K 50
python  main_FedNSAM.py --alg FedNesterov --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedNSAM --lr_decay 2 --gamma 0.85  --CNN   resnet18 --E 5 --batch_size 50   --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10  --rho 0.01 --lora 0 --K 50
python  main_FedNSAM.py --alg FedNSAM --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedNSAM --lr_decay 2 --gamma 0.85  --CNN   resnet18 --E 5 --batch_size 50   --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10  --rho 0.01 --lora 0 --K 50

```

* 这里解释一下 --num_gpus_per 0.1的意思是如果你用的是4090显卡24g显存，那么你每个客户端将分配0.1张显卡，即2.4g显存。
* --lr_decay 2 解释一下，这个是余弦学习率下降
* --gpu 0 是指使用的是第0块gpu（gpu序号）
* --alpha_value 0.1 是迪利克雷非立同分布常数
* --alpha_value 1 这个时候是iid情况
* --lora 0 是否使用lora微调，从头训练的情况下，不用lora微调 选0就行
* --normalization BN resnet的归一化层，我选的是BN层，这个效果更好，选择GN也行，收敛的慢
* --data_name timy imagenet数据集需要自己下载，网址在下面

---
## 联邦大模型微调 vit


```bash
python  main_FedNSAM.py --alg FedNSAM --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 101  --extname FedNSAM --lr_decay 2 --gamma 0.5  --CNN   VIT-B --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.2 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 50 --preprint 10  --rho 0.05 --lora 1 --K 50
```

* --lora 1 使用lora微调
* --batch_size 16 显存限制原因，16效果还可以
* --num_gpus_per 0.2 五个客户端，每个客户端使用0.2张卡
* --lr 1e-3 这个学习率微调lora最好

下载模型权重网址：
下载下来的权重直接放主文件夹下面就行，你也可以自己该目类

vit-base：
https://huggingface.co/Junkang2/vit/tree/main

swin_transformer 
https://huggingface.co/Junkang2/swin_transformer/tree/main

## Dataset

数据集下载网址

Tiny-ImageNet：
https://huggingface.co/datasets/Junkang2/Tiny-ImageNet/upload/main

The code supports multiple datasets:

* **CIFAR-10 / CIFAR-100**
* **Tiny-ImageNet**

## 🤖 **大语言模型训练示例（RoBERTa-base + GLUE-SST2）**
```bash
python new_llm.py \
  --alg FedAdamW \
  --lr 2e-4 \
  --data_name sst2 \
  --alpha_value 0.8 \
  --alpha 0.9 \
  --epoch 101 \
  --extname RoBERTa_SST2 \
  --lr_decay 2 \
  --gamma 0.9 \
  --CNN roberta_base \
  --E 10 \
  --batch_size 16 \
  --gpu 0 \
  --p 1 \
  --num_gpus_per 0.25 \
  --selection 0.2 \
  --pre 1 \
  --num_workers 20 \
  --preprint 5 \
  --K 50 \
  --r 16 \
  --lora 1 \
  --print 1
```
数据集和模型权重下载地址：
* RoBERTa_base模型权重下载地址，下载完之后放入 roberta_base 文件夹即可。
https://huggingface.co/FacebookAI/roberta-base/tree/main

* 数据集下载地址在hugging face上
  sst2 https://huggingface.co/datasets/SetFit/sst2/tree/main
 全部数据集下载地址：
https://huggingface.co/datasets/Junkang2/glue/tree/main


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
