import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from filelock import FileLock
import numpy as np
from torch.utils.data import SubsetRandomSampler
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import time
import random
from math import exp
from copy import deepcopy
import ray
import argparse
from torchsummary import summary
from tensorboardX import SummaryWriter
from dirichlet_data import data_from_dirichlet
from torch.autograd import Variable

from dirichlet_data_1 import data_from_pathological
#from optimizer.GSAM import GSAM
from optimizer.Nesterov import Nesterov
from sam import SAM
from optimizer import LESAM
from torchvision.models import vgg11
#import models

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
from models.resnet import ResNet18,ResNet50,ResNet10
from models.resnet_bn import ResNet18BN,ResNet50BN,ResNet10BN,ResNet34BN
os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lg', default=1.0, type=float, help='learning rate')
parser.add_argument('--epoch', default=1001, type=int, help='number of epochs to train')
parser.add_argument('--num_workers', default=100, type=int, help='#workers')
parser.add_argument('--batch_size', default=50, type=int, help='# batch_size')
parser.add_argument('--E', default=5, type=int, help='# batch_size')
parser.add_argument('--alg', default='FedLESAM', type=str, help='FedAvg')  # FedMoment cddplus cdd SCAF atte
parser.add_argument('--extname', default='EM', type=str, help='extra_name')
parser.add_argument('--gpu', default='0', type=str, help='use which gpus')
parser.add_argument('--lr_decay', default='0.998', type=float, help='lr_decay')
parser.add_argument('--data_name', default='CIFAR100', type=str, help='imagenet,CIFAR100')
parser.add_argument('--tau', default='0.01', type=float, help='only for FedAdam ')
parser.add_argument('--lr_ps', default='1', type=float, help='only for FedAdam ')
parser.add_argument('--alpha_value', default='0.1', type=float, help='for dirichlet')
parser.add_argument('--selection', default='0.1', type=float, help=' C')
parser.add_argument('--check', default=0, type=int, help=' if check')
parser.add_argument('--T_part', default=10, type=int, help=' for mom_step')
parser.add_argument('--alpha', default=0.01, type=float)
parser.add_argument('--CNN', default='lenet5', type=str)
parser.add_argument('--gamma', default=0.85, type=float)
parser.add_argument('--p', default=10, type=float)
parser.add_argument('--freeze-layers', type=bool, default=False)
parser.add_argument('--datapath', type=str, default="./data")
parser.add_argument('--num_gpus_per', default=1, type=float)
parser.add_argument('--normalization', default='BN', type=str)
parser.add_argument('--pre', default=1, type=int)
parser.add_argument('--print', default=0, type=int)

parser.add_argument("--rho", type=float, default=0.05, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--adaptive", type=bool, default=True, help="True if you want to use the Adaptive SAM.")
parser.add_argument("--R", type=int, default=1, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--optimizer', default='SGD', type=str, help='adam')
parser.add_argument("--preprint", type=int, default=10, help="")
parser.add_argument("--clf", type=int, default=10, help="")
parser.add_argument('--method', default= 'pathological', type=str, help='adam')

args = parser.parse_args()
gpu_idx = args.gpu
print('gpu_idx', gpu_idx)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
print(torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
num_gpus_per = args.num_gpus_per  # num_gpus_per = 0.16

num_gpus = len(gpu_idx.split(','))
# num_gpus_per = 1
data_name = args.data_name
CNN = args.CNN
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)

if data_name == 'imagenet':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
import dataset as local_datasets

if data_name == 'imagenet':
    train_dataset = local_datasets.TinyImageNetDataset(
        root=os.path.join(args.datapath, 'tiny-imagenet-200'),
        split='train',
        transform=transform_train
    )

if data_name == 'CIFAR10':
    train_dataset = datasets.CIFAR10(
        "./data",
        train=True,
        download=False,
        transform=transform_train)
elif data_name == 'EMNIST':
    train_dataset = datasets.EMNIST(
        "./data",
        split='byclass',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,)),
        ])
    )

elif data_name == 'CIFAR100':
    train_dataset = datasets.cifar.CIFAR100(
        "./data",
        train=True,
        download=True,
        transform=transform_train
    )
elif data_name == 'MNIST':
    train_dataset = datasets.EMNIST(
        "./data",
        # split='mnist',
        split='balanced',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,)),
        ])
    )





def get_data_loader(pid, data_idx, batch_size, data_name):
    """Safely downloads data. Returns training/validation set dataloader. """
    sample_chosed = data_idx[pid]
    train_sampler = SubsetRandomSampler(sample_chosed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler, num_workers=0,generator=torch.Generator().manual_seed(42))
        #sampler = train_sampler, num_workers = 0)
    return train_loader

def get_data_loader_test(data_name):
    """Safely downloads data. Returns training/validation set dataloader."""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if data_name == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_dataset = local_datasets.TinyImageNetDataset(
            root=os.path.join(args.datapath, 'tiny-imagenet-200'),
            split='test',
            transform=transform_train
        )
    if data_name == 'CIFAR10':
        test_dataset = datasets.CIFAR10("./data", train=False, transform=transform_test)
    elif data_name == 'EMNIST':
        test_dataset = datasets.EMNIST("./data", split='byclass', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))]))
    elif data_name == 'CIFAR100':
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, transform=transform_test
                                             )
    elif data_name == 'MNIST':
        # test_dataset = datasets.EMNIST("./data",split='mnist', train=False, transform=transforms.Compose([
        test_dataset = datasets.EMNIST("./data", split='balanced', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=200,
        shuffle=False,
        num_workers=4)
    return test_loader


def get_data_loader_train(data_name):
    """Safely downloads data. Returns training/validation set dataloader."""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if data_name == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_dataset = local_datasets.TinyImageNetDataset(
            root=os.path.join(args.datapath, 'tiny-imagenet-200'),
            split='train',
            transform=transform_train
        )
    if data_name == 'CIFAR10':
        test_dataset = datasets.CIFAR10("./data", train=True, transform=transform_test)
    elif data_name == 'EMNIST':
        test_dataset = datasets.EMNIST("./data", split='byclass', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))]))
    elif data_name == 'CIFAR100':
        test_dataset = datasets.cifar.CIFAR100("./data", train=True, transform=transform_test)
    elif data_name == 'MNIST':
        # test_dataset = datasets.EMNIST("./data",split='mnist', train=False, transform=transforms.Compose([
        test_dataset = datasets.EMNIST("./data", split='balanced', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=200,
        shuffle=False,
        num_workers=4)
    return test_loader


if data_name == 'imagenet':
    def evaluate(model, test_loader, train_loader):
        """Evaluates the accuracy of the model on a validation dataset."""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100. * correct / total, torch.tensor(0), torch.tensor(0)
else:
    def evaluate(model, test_loader, train_loader):
        """Evaluates the accuracy of the model on a validation dataset."""
        criterion = nn.CrossEntropyLoss()
        model.eval()
        correct = 0
        total = 0
        test_loss = 0
        train_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                test_loss += criterion(outputs, target)

            for batch_idx, (data, target) in enumerate(train_loader):
                data_train = data.to(device)
                target_train = target.to(device)
                outputs_train = model(data_train)
                train_loss += criterion(outputs_train, target_train)
        return 100. * correct / total, test_loss / len(test_loader), train_loss / len(train_loader)


class ConvNet_MNIST(nn.Module):
    """TF Tutorial for EMNIST."""

    def __init__(self):
        super(ConvNet_MNIST, self).__init__()
        self.fc1 = nn.Linear(784, 47)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = self.sigmoid(x)
        # return F.log_softmax(x, dim=1)
        return x

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


# '''

import torch.nn as nn
import torchvision.models as models




class SCAFNET(nn.Module):
    """TF Tutorial for EMNIST."""

    def __init__(self):
        super(SCAFNET, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 62)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

class ResNet18pre(nn.Module):
  def __init__(self, num_classes=10, l2_norm=False):
    super(ResNet18pre, self).__init__()
    if args.pre==1:
      resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
      resnet18 = models.resnet18()
    resnet18.fc = nn.Linear(512,  num_classes)
    self.model = resnet18

  def forward(self, x):
    x = self.model(x)
    return x

  def get_weights(self):
    return {k: v.cpu() for k, v in self.state_dict().items()}

  def set_weights(self, weights):
    self.load_state_dict(weights)

  def get_gradients(self):
    grads = []
    for p in self.parameters():
        grad = None if p.grad is None else p.grad.data.cpu().numpy()
        grads.append(grad)
    return grads

  def set_gradients(self, gradients):
      for g, p in zip(gradients, self.parameters()):
          if g is not None:
              p.grad = torch.from_numpy(g)

class ResNet50pre(nn.Module):
  def __init__(self, num_classes=10, l2_norm=False):
    super(ResNet50pre, self).__init__()
    if args.pre==1:
      resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
      resnet50 = models.resnet50()
    resnet50.fc = nn.Linear(2048,  num_classes)
    #nn.Linear(2048, 100)
    self.model = resnet50

  def forward(self, x):
    x = self.model(x)
    return x

  def get_weights(self):
    return {k: v.cpu() for k, v in self.state_dict().items()}

  def set_weights(self, weights):
    self.load_state_dict(weights)

  def get_gradients(self):
    grads = []
    for p in self.parameters():
        grad = None if p.grad is None else p.grad.data.cpu().numpy()
        grads.append(grad)
    return grads

  def set_gradients(self, gradients):
      for g, p in zip(gradients, self.parameters()):
          if g is not None:
              p.grad = torch.from_numpy(g)



import torch.nn as nn
import torchvision.models as models


def blockVGG(covLayerNum, inputChannel, outputChannel, kernelSize, withFinalCov1: bool):
    layer = nn.Sequential()
    layer.add_module('conv2D1', nn.Conv2d(inputChannel, outputChannel, kernelSize, padding=1))
    layer.add_module('relu-1', nn.ReLU())
    for i in range(covLayerNum - 1):
        layer.add_module('conv2D{}'.format(i), nn.Conv2d(outputChannel, outputChannel, kernelSize, padding=1))
        layer.add_module('relu{}'.format(i), nn.ReLU())
    if withFinalCov1:
        layer.add_module('Conv2dOne', nn.Conv2d(outputChannel, outputChannel, 1))
    layer.add_module('max-pool', nn.MaxPool2d(2, 2))
    return layer

# VGG11
# '''
class VGG11_10(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = blockVGG(1, 3, 64, 3, False)

        self.layer2 = blockVGG(1, 64, 128, 3, False)

        self.layer3 = blockVGG(2, 128, 256, 3, False)

        self.layer4 = blockVGG(2, 256, 512, 3, False)

        self.layer5 = blockVGG(2, 512, 512, 3, False)
        self.layer6 = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            # nn.ReLU(),
            # nn.Softmax(1)
        )

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.shape[0], -1)
        x = self.layer6(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


class VGG11_100(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = blockVGG(1, 3, 64, 3, False)

        self.layer2 = blockVGG(1, 64, 128, 3, False)

        self.layer3 = blockVGG(2, 128, 256, 3, False)

        self.layer4 = blockVGG(2, 256, 512, 3, False)

        self.layer5 = blockVGG(2, 512, 512, 3, False)
        self.layer6 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            # nn.ReLU(),
            # nn.Softmax(1)
        )

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.shape[0], -1)
        x = self.layer6(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


# Lenet5
class Lenet5_10(nn.Module):
    """TF Tutorial for CIFAR."""

    def __init__(self):
        super(Lenet5_10, self).__init__()
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


class Lenet5_100(nn.Module):

    def __init__(self):
        super(Lenet5_100, self).__init__()
        self.n_cls = 100
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


from torch import nn
import math


class VGG(nn.Module):
    def __init__(self, features, num_classes=100, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # nn.Linear(512 * 2 * 2, 4096),
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4096, 4096),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4096, num_classes)
            nn.Linear(512, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(True)]
            else:
                layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}


from torch import nn
from torchvision import datasets, transforms, models

if CNN == 'resnet34':
    def ConvNet(num_classes=10):
        return ResNet34BN(num_classes=10)
    def ConvNet100(num_classes=100):
        return ResNet34BN(num_classes=100)
    def ConvNet200(num_classes=200):
        return ResNet34BN(num_classes=200)

if CNN == 'resnet10':
    if args.normalization=='BN':
        def ConvNet(num_classes=10):
            return ResNet10BN(num_classes=10)
        def ConvNet100(num_classes=100):
            return ResNet10BN(num_classes=100)
        def ConvNet200(num_classes=200):
            return ResNet10BN(num_classes=200)
    if args.normalization=='GN':
        def ConvNet(num_classes=10):
            return ResNet10(num_classes=10)
        def ConvNet100(num_classes=100):
            return ResNet10(num_classes=100)
        def ConvNet200(num_classes=200):
            return ResNet10(num_classes=200)


if CNN == 'resnet50':
    def ConvNet(num_classes=10):
        return ResNet50BN(num_classes=10)
    def ConvNet100(num_classes=100):
        return ResNet50BN(num_classes=100)
    def ConvNet200(num_classes=200):
        return ResNet50BN(num_classes=200)


if CNN == 'resnet18':
    if args.normalization == 'BN':
        def ConvNet(num_classes=10, l2_norm=False):
            return ResNet18BN(num_classes=10)
        def ConvNet100(num_classes=100, l2_norm=False):
            return ResNet18BN(num_classes=100)
        def ConvNet200(num_classes=200, l2_norm=False):
            return ResNet18BN(num_classes=200)
    if args.normalization=='GN':
        def ConvNet(num_classes=10):
            return ResNet18(num_classes=10)
        def ConvNet100(num_classes=100):
            return ResNet18(num_classes=100)
        def ConvNet200(num_classes=200):
            return ResNet18(num_classes=200)


if CNN == 'resnet18pre':
    def ConvNet(num_classes=10):
        return ResNet18pre(num_classes=10)
    def ConvNet100(num_classes=100):
        return ResNet18pre(num_classes=100)
    def ConvNet200(num_classes=200):
        return ResNet18pre(num_classes=200)



if CNN == 'resnet50pre':
    def ConvNet(num_classes=10):
        return ResNet50pre(num_classes=10)
    def ConvNet100(num_classes=100):
        return ResNet50pre(num_classes=100)
    def ConvNet200(num_classes=200):
        return ResNet50pre(num_classes=200)

if CNN == 'vgg11':
    def ConvNet():
        return VGG11_10()


    def ConvNet100():
        return VGG11_100()

if CNN == 'lenet5':
    def ConvNet():
        return Lenet5_10()


    def ConvNet100():
        return Lenet5_100()

class ConvNet_MNIST(nn.Module):
    """TF Tutorial for EMNIST."""

    def __init__(self):
        super(ConvNet_MNIST, self).__init__()
        self.fc1 = nn.Linear(784, 47)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


@ray.remote
class ParameterServer(object):
    def __init__(self, lr, alg, tau, selection, data_name, num_workers):
        if data_name == 'CIFAR10':
            self.model = ConvNet()
        elif data_name == 'EMNIST':
            self.model = SCAFNET()
        elif data_name == 'CIFAR100':
            self.model = ConvNet100()
        elif data_name == 'MNIST':
            self.model = ConvNet_MNIST()
        if data_name == 'imagenet':
            self.model = ConvNet200()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.momen_v ={}
        self.momen_v_hat = {}
        self.c_all = None
        self.c_all_pre = None
        self.gamma = args.gamma
        # self.gamma = 0.9
        self.beta = 0.99  
        self.alg = alg
        self.num_workers = num_workers
        self.lr_ps = lr
        self.lg = 1.0
        self.ps_c = None
        self.tau = tau
        self.selection = selection
        self.cnt = 0
        self.alpha = args.alpha
        self.h = {}
        self.momen_m = {}
        self.t=0


    def apply_weights_avg(self, num_workers, *weights):

        ps_w = self.model.get_weights()  # w : ps_w
        sum_weights = {}  # delta_w : sum_weights
        global_weights = {}
        for weight in weights:
            for k, v in weight.items():
                if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                    sum_weights[k] += v / (num_workers * self.selection)
                else:
                    sum_weights[k] = v / (num_workers * self.selection)
        for k, v in sum_weights.items():  # w = w + delta_w
            global_weights[k] = ps_w[k] + sum_weights[k]
        self.model.set_weights(global_weights)
        return self.model.get_weights()


    def apply_weights_moment(self, num_workers, *weights):
        self.gamma = 0.9
        sum_weights = {}
        for weight in weights:
            for k, v in weight.items():
                if k in sum_weights.keys():
                    sum_weights[k] += v / (num_workers * self.selection)
                else:
                    sum_weights[k] = v / (num_workers * self.selection)
        weight_ps = self.model.get_weights()
        if self.momen_v=={}:
            self.momen_v = deepcopy(sum_weights)
        else:
            for k, v in self.momen_v.items():
                self.momen_v[k] = self.gamma * v + sum_weights[k]
        seted_weight = {}
        for k, v in weight_ps.items():
            seted_weight[k] = v + self.momen_v[k]
        self.model.set_weights(seted_weight)
        return self.model.get_weights()


    def apply_weights_FedDyn(self, num_workers, *weights):
        sum_weights = {}
        for weight in weights:
            for k, v in weight.items():
                if k in sum_weights.keys():
                    sum_weights[k] += v / (num_workers * self.selection)
                else:
                    sum_weights[k] = v / (num_workers * self.selection)
        weight_ps = self.model.get_weights()
        if self.momen_v=={}:
            self.momen_v = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        #else:
        for k, v in self.momen_v.items():
            self.momen_v[k] = self.momen_v[k] -args.alpha* self.selection*sum_weights[k]
        seted_weight = {}
        for k, v in weight_ps.items():
            seted_weight[k] = v - (1/args.alpha)*self.momen_v[k]+ sum_weights[k]
        self.model.set_weights(seted_weight)
        return self.model.get_weights()



    def apply_weights_adam(self, num_workers, *weights):
        self.beta = 0.99
        delta_t = {}
        for weight in weights:
            for k, v in weight.items():
                if k in delta_t.keys():
                    delta_t[k] += v / (num_workers * selection)
                else:
                    delta_t[k] = v / (num_workers * selection)
        weight_ps = self.model.get_weights()
        if  self.momen_m=={}:
            for k, v in delta_t.items():
                #self.momen_m[k] = 0.1 * delta_t[k]
                self.momen_m[k] = delta_t[k]
        else:
            for k, v in delta_t.items():
                self.momen_m[k] = 0.9 * self.momen_m[k] + 0.1 * delta_t[k]

        if self.momen_v=={}:
            self.momen_v = deepcopy(delta_t)
            for k, v in delta_t.items():
                #self.momen_v[k] = (1 - self.beta) * v.mul(v)
                self.momen_v[k] =v.mul(v)

        else:
            for k, v in self.momen_v.items():
                self.momen_v[k] = self.beta * v + (1 - self.beta) * delta_t[k].mul(delta_t[k])
        seted_weight = {}
        for k, v in weight_ps.items():
            seted_weight[k] = v + args.lr_ps * self.momen_m[k] / (self.momen_v[k].sqrt() + self.tau)

        self.model.set_weights(seted_weight)
        return self.model.get_weights()


    def apply_weights_avgACG(self, num_workers, *weights):
        self.gamma = 0.85
        ps_w = self.model.get_weights()
        sum_weights = {}
        for weight in weights:
            for k, v in weight.items():
                if k in sum_weights.keys():
                    sum_weights[k] += 1 / (self.num_workers * self.selection) * v
                else:
                    sum_weights[k] = 1 / (self.num_workers * self.selection) * v
        if  self.momen_v=={}:
            self.momen_v = deepcopy(sum_weights)
        else:
            for k, v in self.momen_v.items():
                self.momen_v[k] = self.gamma * v + sum_weights[k]
        seted_weight = {}
        for k, v in ps_w.items():
            seted_weight[k] = v + self.momen_v[k]
        self.model.set_weights(seted_weight)
        return self.model.get_weights(), self.momen_v


    def apply_weights_FedNesterov(self, num_workers, *weights):
        self.gamma = 0.85
        ps_w = self.model.get_weights()
        sum_weights = {}
        for weight in weights:
            for k, v in weight.items():
                if k in sum_weights.keys():
                    sum_weights[k] += 1 / (self.num_workers * self.selection) * v
                else:
                    sum_weights[k] = 1 / (self.num_workers * self.selection) * v
        if self.momen_v=={}:
            self.momen_v = deepcopy(sum_weights)
        else:
            for k, v in self.momen_v.items():
                self.momen_v[k] = self.gamma * v + sum_weights[k]
        seted_weight = {}
        for k, v in ps_w.items():
            seted_weight[k] = v + self.momen_v[k]
        self.model.set_weights(seted_weight)
        return self.model.get_weights(), self.momen_v


    def load_dict(self):
        self.func_dict = {
            'FedAvg': self.apply_weights_avg,
            'FedMoment': self.apply_weights_moment,
            'SCAFFOLD': self.apply_weights_avg,
            'IGFL': self.apply_weights_avg,
            'FedAdam': self.apply_weights_adam,
            'FedCM': self.apply_weights_avg,
            'FedDC': self.apply_weights_avg,
            'FedSAM': self.apply_weights_avg,
            'MoFedSAM': self.apply_weights_avg,
            'Fedprox': self.apply_weights_avg,
            'FedACG': self.apply_weights_avgACG,
            'FedNesterov': self.apply_weights_FedNesterov,
            'FedFree': self.apply_weights_avg,
            'FedAdamL': self.apply_weights_avg,
            'FAFED': self.apply_weights_avg,
            'FedLion': self.apply_weights_avg,
            'FedAdamP': self.apply_weights_avg,
            'FedNadam': self.apply_weights_avgACG,
            'FedCM+': self.apply_weights_avg,
            'FedANAG': self.apply_weights_avg,
            'FedDyn': self.apply_weights_FedDyn,
            'FedSMOO': self.apply_weights_FedDyn,
            'SCAFFOLD+': self.apply_weights_avg,
            'FedSWAS': self.apply_weights_avg,
            'FedGAMMA': self.apply_weights_avg,
            'FedSWA': self.apply_weights_avg,
            'FedNSAM': self.apply_weights_avgACG,
            #'FedDyn':  self.apply_weights_avg,

        }

    def apply_weights_func2(self, alg, num_workers, weights, ps_c):
        self.load_dict()

        return self.func_dict.get(alg, None)(num_workers, weights, ps_c)

    def apply_weights_func(self, alg, num_workers, *weights):
        self.load_dict()
        return self.func_dict.get(alg, None)(num_workers, *weights)

    def apply_ci(self, alg, num_workers, *cis):
        args.gamma = 0.2
        sum_c = {}  # delta_c :sum_c
        for ci in cis:
            for k, v in ci.items():
                if k in sum_c.keys():
                    sum_c[k] += v / (num_workers * selection)
                else:
                    sum_c[k] = v / (num_workers * selection)
        if self.ps_c == None:
            self.ps_c = sum_c
            return self.ps_c
        for k, v in self.ps_c.items():
            if alg in {'FedSAMC', 'FedSAMS'}:
                args.gamma = 0.1
                self.ps_c[k] = (1 - args.gamma) * self.ps_c[k] + args.gamma * sum_c[k]
            if alg in {'FedSTORM', 'FedNesterov', 'FedLESAM', 'FedSAMSM', 'FedLion', 'FedAdamP'}:
                self.ps_c[k] = sum_c[k]
            if alg in {'IGFL_prox'}:
                self.ps_c[k] = v * args.gamma + sum_c[k]
            if alg in {'IGFL_prox', 'FedAGM', 'IGFL', 'MoFedSAM', 'stem', 'FedPGN', 'FedFree', 'FedCM_VR',
                        'FedAdamL','FedCM', 'FAFED','FedCM+', 'SCAFFOLD+'}:
                self.ps_c[k] = v + sum_c[k]
            if alg in { 'SCAFFOLD+'}:
                self.ps_c[k] = v + sum_c[k]*args.gamma
            if alg in { 'SCAFFOLD'}:
                self.ps_c[k] = v + sum_c[k]*selection
            if alg in { 'FedANAG'}:
                self.ps_c[k] = v +v*0.9/1.9 + sum_c[k]
            else:
                self.ps_c[k] = v + sum_c[k] * args.gamma
        return self.ps_c

    def get_weights(self):
        return self.model.get_weights()

    def get_ps_c(self):
        return self.ps_c

    def get_state(self):
        return self.ps_c, self.c_all

    def set_state(self, c_tuple):
        self.ps_c = c_tuple[0]
        self.c_all = c_tuple[1]

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_attention(self):
        return self.alpha




@ray.remote(num_gpus=num_gpus_per)
class DataWorker(object):

    def __init__(self, pid, data_idx, num_workers, lr, batch_size, alg, data_name, selection, T_part):
        self.alg = alg
        if data_name == 'CIFAR10':
            self.model = ConvNet().to(device)
        elif data_name == 'CIFAR100':
            self.model = ConvNet100().to(device)
        elif data_name == 'MNIST':
            self.model = ConvNet_MNIST().to(device)
        if data_name == 'imagenet':
            self.model = ConvNet200().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.num_workers = num_workers
        self.data_iterator = None
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.loss = 0
        self.lr_decay = lr_decay
        self.alg = alg
        self.data_idx = data_idx
        self.flag = False
        self.ci = None
        self.selection = selection
        self.T_part = T_part
        self.Li = None
        self.hi = None
        self.momen_v = {}
        self.momen_m = {}
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.old = {}
        self.t = 0

    def data_id_loader(self, index):
        self.data_iterator = get_data_loader(index, self.data_idx, batch_size, data_name)

    def state_id_loader(self, index):
        if not c_dict.get(index):
            return
        self.ci = c_dict[index]

    def state_mi_loader(self, index):
        if not mi_dict.get(index):
            return
        self.momen_m = mi_dict[index]

    def state_vi_loader(self, index):
        if not vi_dict.get(index):
            return
        self.momen_v = vi_dict[index]

    def state_ti_loader(self, index):
        if not ti_dict.get(index):
            return
        self.t = ti_dict[index]

    def state_hi_loader(self, index):
        if not hi_dict.get(index):
            return
        self.hi = hi_dict[index]

    def state_Li_loader(self, index):
        if not Li_dict.get(index):
            return
        self.Li = Li_dict[index]

    def get_train_loss(self):
        return self.loss

    def update_fedavg(self, weights, E, index, lr):
        self.model.set_weights(weights)
        self.model.to(device)
        self.data_id_loader(index)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        if args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-3)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        # self.loss = loss.item()
        delta_w = {}
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w


    def update_Fedprox(self, weights, E, index, lr):
        self.model.set_weights(weights)
        self.data_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                ce_loss = self.criterion(output, target)
                loss_cg = 0
                alpha = args.alpha = 0.01
                for n, p in model.named_parameters():
                    weights[n] = weights[n].to(device)
                    L1 = alpha / 2 * torch.sum((p - weights[n]) * (p - weights[n]))
                    loss_cg += L1.item()
                loss = ce_loss + loss_cg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        self.loss = loss.item()
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w



    def update_scafplus(self, weights, E, index, ps_c,lr):
        self.model.set_weights(weights)  # y_i = x, x:weights
        if self.ci == None:  # ci_0 = 0 , c = 0
            self.ci =  {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        if ps_c == None:
            ps_c = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        self.data_id_loader(index)
        self.state_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
        self.model.to(device)
        for k, v in self.model.named_parameters():
             ps_c[k]=ps_c[k].to(device)
             self.ci[k]=self.ci[k].to(device)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                for k, v in self.model.named_parameters():
                    v.grad.data = v.grad.data + ps_c[k]-self.ci[k]
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()  # y_i = y_i-lr*g
        #ci={}
        for k, v in self.model.named_parameters():
             ps_c[k]=ps_c[k].to('cpu')
             self.ci[k]=self.ci[k].to('cpu')
             #ci[k]=self.ci[k].to('cpu')
        send_ci = {}
        for k, v in self.model.get_weights().items():
            if k not in self.ci.keys():
                continue
            self.ci[k] =( weights[k]-v) / (E * len(self.data_iterator)*lr) +self.ci[k].to('cpu')-ps_c[k].to('cpu')
        delta_w = {}
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        for k, v in self.model.get_weights().items():
            send_ci[k] = -ps_c[k] +self.ci[k]
            #send_ci[k] = -ci[k] + self.ci[k]
        c_dict[index] = deepcopy(self.ci)                       
        return delta_w, send_ci

    def update_scafplus2(self, weights, E, index, ps_c, lr):
        self.model.set_weights(weights)  # y_i = x, x:weights
        num_workers = self.num_workers
        if self.ci == None:  # ci_0 = 0 , c = 0
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = deepcopy(zero_weight)
        if ps_c == None:
            ps_c = deepcopy(zero_weight)
        self.data_id_loader(index)
        self.state_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()  # y_i = y_i-lr*g
                new_weights = deepcopy(self.model.get_weights())
                for k, v in self.model.get_weights().items():
                    new_weights[k] = v - (-self.ci[k] + ps_c[k])  # y_i = y_i -lr*(-ci + c)
                self.model.set_weights(new_weights)
        send_ci = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            self.ci[k] = (weights[k] - v) / (E * len(self.data_iterator)) + self.ci[k] - ps_c[k]
        self.loss = loss.item()
        for k, v in self.model.get_weights().items():
            send_ci[k] = -ps_c[k] + self.ci[k]
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        c_dict[index] = deepcopy(self.ci)
        return delta_w, send_ci


    def update_FedLESAM(self, weights, E, index, ps_c,lr):
        self.model.set_weights(weights)  # y_i = x, x:weights
        self.data_id_loader(index)
        self.state_id_loader(index)
        base_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
        self.optimizer = LESAM(self.model.parameters(), base_optimizer, rho=0.01)

        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                self.optimizer.paras = [data , target, self.criterion , self.model]
                self.optimizer.step(ps_c)
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)
                base_optimizer.step()
        delta_w = {}
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w,delta_w


    def update_FedGAMMA(self, weights, E, index, ps_c,lr):
        self.model.set_weights(weights)  # y_i = x, x:weights
        if self.ci == None:
            self.ci = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        if ps_c == None:
            ps_c = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        self.data_id_loader(index)
        self.state_id_loader(index)
        base_optimizer = torch.optim.SGD
        self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr, momentum=0,rho=0.05)
        for n, p in model.named_parameters():
            ps_c[n] = ps_c[n].to(device)
            self.ci[n] = self.ci[n].to(device)
            weights[n] = weights[n].to(device)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                lg_loss = 0
                loss_c = self.criterion(output, target)
                for n, p in model.named_parameters():
                    lossh = (p * (-self.ci[n] + ps_c[n])).sum()
                    lg_loss += lossh.item()
                loss = loss_c + lg_loss
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                # loss_function(output,model(input)).backward()
                self.criterion(self.model(data), target).backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.second_step(zero_grad=True)
        send_ci = deepcopy(self.model.get_weights())
        ci=deepcopy(self.ci)
        for k, v in self.model.get_weights().items():
            ps_c[k] = ps_c[k].to('cpu')
            self.ci[k] = self.ci[k].to('cpu')
            weights[k]=weights[k].to('cpu')
            ci[k]=ci[k].to('cpu')
            self.ci[k] =( weights[k]-v) / (E * len(self.data_iterator)*lr) +ci[k]-ps_c[k]
        self.loss = loss.item()
        for k, v in self.model.get_weights().items():
            send_ci[k] = -ci[k] + self.ci[k]
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        c_dict[index] = deepcopy(self.ci)
        return delta_w, send_ci


    def update_fedDyn(self, weights, E, index, lr):
        self.model.set_weights(weights)
        if self.hi ==None:
            self.hi = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        self.data_id_loader(index)
        self.state_id_loader(index)
        self.state_hi_loader(index)
        alpha=args.alpha
        for n, p in model.named_parameters():
            self.hi[n] = self.hi[n].to(device)
            weights[n] = weights[n].to(device)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                ce_loss = self.criterion(output, target)
                reg_loss = 0
                lg_loss=0
                for n, p in model.named_parameters():
                    lossh=torch.sum((p*(-self.hi[n])))
                    l1=torch.sum((p-weights[n])**2)*args.alpha*0.5
                    lg_loss +=l1.item()
                    reg_loss += lossh.item()
                loss = ce_loss+reg_loss+lg_loss
                #for n, p in model.named_parameters():
                #    lossh=torch.sum((p*(-self.hi[n]+weights[n])))
                #    reg_loss += lossh.item()
                #loss = ce_loss +args.alpha* reg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        for k, v in self.model.named_parameters():
            self.hi[k] = self.hi[k].to('cpu')

        for k, v in self.model.get_weights().items():
            if k not in self.hi.keys():
                continue
            weights[k] = weights[k].to('cpu')
            self.hi[k] = self.hi[k] - (v - weights[k])*args.alpha
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            weights[k] = weights[k].to('cpu')
            delta_w[k] = v - weights[k]
        hi_dict[index] = deepcopy(self.hi)
        return delta_w

    def update_FedSWA(self, weights, E, index,lr):
        self.model.set_weights(weights)
        self.data_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
        a1=lr
        a2=args.rho*lr
        i=0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                i=i+1
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
                #lr=lr*0.98
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                lr=(1-i/(len(self.data_iterator)*E))*a1+(i/(len(self.data_iterator)*E))*a2
        self.loss = loss.item()
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w

    def update_FedSWAS(self, weights, E, index, ps_c,lr):
        self.model.set_weights(weights)  # y_i = x, x:weights
        if self.ci == None:  # ci_0 = 0 , c = 0
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = deepcopy(zero_weight)
        if ps_c == None:
            ps_c = deepcopy(zero_weight)
        self.data_id_loader(index)
        self.state_id_loader(index)
        a1=lr
        a2=args.rho*lr
        i=0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
                i=i+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()  # y_i = y_i-lr*g
                new_weights = deepcopy(self.model.get_weights())
                for k, v in self.model.get_weights().items():
                    new_weights[k] = v - (-self.ci[k] +ps_c[k])*lr  # y_i = y_i -lr*(-ci + c)
                self.model.set_weights(new_weights)
                lr=(1-i/(len(self.data_iterator)*E))*a1+(i/(len(self.data_iterator)*E))*a2
        lr=(a1+a2)/2
        send_ci = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            self.ci[k] =( weights[k]-v) / (E * len(self.data_iterator)*lr) +self.ci[k]-ps_c[k]
        self.loss = loss.item()
        for k, v in self.model.get_weights().items():
            send_ci[k] = -ps_c[k] + self.ci[k]
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        ci_copy = deepcopy(self.ci)
        c_dict[index] = ci_copy                      
        return delta_w, send_ci

    def update_MoFedSAM(self, weights, E, index, ps_c,lr):
        self.model.set_weights(weights)
        if ps_c is None:
            ps_c = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        self.data_id_loader(index)
        self.gamma = 0.9
        base_optimizer = torch.optim.SGD
        self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr, momentum=0,rho=0.05)
        for k, v in self.model.named_parameters():
             ps_c[k]=ps_c[k].to(device)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                self.criterion(self.model(data), target).backward()
                for k, v in self.model.named_parameters():
                    v.grad.data=(1 - self.gamma) * v.grad.data+self.gamma * ps_c[k]
                self.optimizer.second_step(zero_grad=True)
        send_ci = {}
        for k, v in self.model.named_parameters():
             ps_c[k]=ps_c[k].to('cpu')
        for k, v in self.model.get_weights().items():
            send_ci[k] = - ps_c[k] -1 / (E * len(self.data_iterator)*lr) * (v - weights[k])
        delta_w = {}
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w, send_ci




    def update_SAM(self, weights, E, index, lr):
        self.model.set_weights(weights)  # y_i = x, x:weights
        self.data_id_loader(index)
        base_optimizer = torch.optim.SGD
        self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr, momentum=0, rho=args.rho)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                # loss_function(output,model(input)).backward()
                self.criterion(self.model(data), target).backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.second_step(zero_grad=True)
        self.loss = loss.item()
        delta_w = {}
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w


    def update_FedSMOO(self, weights, E, index, lr):
        self.model.set_weights(weights)  # y_i = x, x:weights
        self.data_id_loader(index)
        if self.hi == None:
            self.hi = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 + args.alpha)
        base_optimizer = torch.optim.SGD
        self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr, momentum=0, rho=args.rho)
        self.data_id_loader(index)
        self.state_id_loader(index)
        self.state_hi_loader(index)
        alpha = args.alpha
        for n, p in model.named_parameters():
            self.hi[n] = self.hi[n].to(device)
            weights[n] = weights[n].to(device)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                ce_loss = self.criterion(output, target)
                reg_loss = 0
                lg_loss = 0
                for n, p in model.named_parameters():
                    lossh = torch.sum((p * (-self.hi[n])))
                    l1 = torch.sum((p - weights[n]) ** 2) * args.alpha * 0.5
                    lg_loss += l1.item()
                    reg_loss += lossh.item()
                loss = ce_loss + reg_loss + lg_loss
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                self.criterion(self.model(data), target).backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.second_step(zero_grad=True)
        for k, v in self.model.named_parameters():
            self.hi[k] = self.hi[k].to('cpu')
        for k, v in self.model.get_weights().items():
            if k not in self.hi.keys():
                continue
            weights[k] = weights[k].to('cpu')
            self.hi[k] = self.hi[k] - (v - weights[k])*args.alpha
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            weights[k] = weights[k].to('cpu')
            delta_w[k] = v - weights[k]
        hi_copy = deepcopy(self.hi)
        hi_dict[index] = hi_copy
        return delta_w



    def update_FedCM(self, weights, E, index, ps_c, lr):
        self.model.set_weights(weights)
        self.model.to(device)
        if ps_c is None:
            ps_c = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        self.data_id_loader(index)
        self.gamma = 0.9
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        for k, v in self.model.named_parameters():
            ps_c[k] = ps_c[k].to(device)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                for k, v in self.model.named_parameters():
                    v.grad.data = (1 - self.gamma) * v.grad.data + self.gamma * ps_c[k]
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        send_ci = {}
        for k, v in self.model.named_parameters():
            ps_c[k] = ps_c[k].to('cpu')
        for k, v in self.model.get_weights().items():
            send_ci[k] = - ps_c[k] - 1 / (E * len(self.data_iterator) * lr) * (v - weights[k])
        delta_w = {}
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w, send_ci


    def update_FedANAG(self, weights, E, index, ps_c, lr):
        self.model.set_weights(weights)
        self.model.to(device)
        if ps_c is None:
            ps_c = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        self.data_id_loader(index)
        self.gamma = 0.9
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        for k, v in self.model.named_parameters():
            ps_c[k] = ps_c[k].to(device)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                for k, v in self.model.named_parameters():
                    v.grad.data = (1 + self.gamma) * v.grad.data + self.gamma**2 * ps_c[k]
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        send_ci = {}
        for k, v in self.model.named_parameters():
            ps_c[k] = ps_c[k].to('cpu')
        for k, v in self.model.get_weights().items():
            send_ci[k] = - ps_c[k] - 1 / (E * len(self.data_iterator) * lr*(1+self.gamma)) * (v - weights[k])
        delta_w = {}
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w, send_ci




    def update_scaf(self, weights, E, index, ps_c, lr):
        self.model.set_weights(weights)
        self.model.to(device)
        if self.ci == None:
            self.ci = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        if ps_c == None:
            ps_c = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        self.data_id_loader(index)
        self.state_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.001)
        for n, p in model.named_parameters():
            ps_c[n] = ps_c[n].to(device)
            self.ci[n] = self.ci[n].to(device)
            weights[n] = weights[n].to(device)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                lg_loss = 0
                loss_c = self.criterion(output, target)
                for n, p in model.named_parameters():
                    lossh = (p * (-self.ci[n] + ps_c[n])).sum()
                    lg_loss += lossh.item()
                loss = loss_c + lg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        send_ci = {}
        ci = deepcopy(self.ci)
        for k, v in self.model.get_weights().items():
            ps_c[k] = ps_c[k].to('cpu')
            self.ci[k] = self.ci[k].to('cpu')
            weights[k] = weights[k].to('cpu')
            ci[k] = ci[k].to('cpu')
            self.ci[k] = (weights[k] - v) / (E * len(self.data_iterator) * lr) + ci[k] - ps_c[k]
        for k, v in self.model.get_weights().items():
            send_ci[k] = -ci[k] + self.ci[k]
        delta_w = {}
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        c_dict[index] = deepcopy(self.ci)
        return delta_w, send_ci


    def update_FedACG(self, weights, E, index, ps_c, lr):
        for k, v in weights.items():
            weights[k] = weights[k] + ps_c[k] * 0.85
        self.model.set_weights(weights)
        self.data_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        if args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-3)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                reg_loss = 0
                for n, p in model.named_parameters():
                    weights[n] = weights[n].to(device)
                    L1 = ((p - weights[n].detach()) ** 2).sum()
                    reg_loss += L1.item()
                loss = self.criterion(output, target)+0.01*reg_loss
                loss.backward()
                self.optimizer.step()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
        for k, v in self.model.get_weights().items():
            weights[k] = weights[k].to('cpu')
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w


    def update_FedDC(self, weights, E, index, ps_c, lr):
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        # print(self.optimizer.state_dict()['param_groups'][0]['lr'])
        num_workers = int(self.num_workers * selection)
        self.model.set_weights(weights)
        # fixed_params = {n: p for n, p in self.model.named_parameters()}

        if self.ci == None:
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = zero_weight
        if ps_c == None:
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            ps_c = zero_weight
        if self.hi == None:
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.hi = zero_weight
        #del zero_weight

        self.data_id_loader(index)
        self.state_id_loader(index)
        self.state_hi_loader(index)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                ce_loss = self.criterion(output, target)
                ## Weight L2 loss
                reg_loss = 0
                loss_cg = 0
                alpha = args.alpha=0.01
                for n, p in model.named_parameters():
                    ps_c[n] = ps_c[n].to(device)
                    self.ci[n] = self.ci[n].to(device)
                    self.hi[n] = self.hi[n].to(device)
                    weights[n] = weights[n].to(device)
                    L1 = alpha / 2 * torch.sum(
                        (p - (weights[n] - self.hi[n])) * (p - (weights[n] - self.hi[n]))) + torch.sum(
                        p * (-self.ci[n] + ps_c[n]))
                    loss_cg += L1.item()
                loss = ce_loss + loss_cg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()

        send_ci = deepcopy(self.model.get_weights())
        ci=deepcopy(self.ci)
        for k, v in self.model.get_weights().items():
            ps_c[k] = ps_c[k].to('cpu')
            self.ci[k] = self.ci[k].to('cpu')
            weights[k]=weights[k].to('cpu')
            ci[k]=ci[k].to('cpu')
            self.ci[k] =( weights[k]-v) / (E * len(self.data_iterator)*lr) +ci[k]-ps_c[k]
        self.loss = loss.item()

        for k, v in self.model.get_weights().items():
            send_ci[k] = -ci[k] + self.ci[k]

        for k, v in self.model.get_weights().items():
            self.hi[k]=self.hi[k].to('cpu')
            self.hi[k] = self.hi[k] + (v - weights[k])
        self.loss = loss.item()
        del ci
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        # ci_copy = deepcopy(self.ci)
        c_dict[index] = deepcopy(self.ci)
        # hi_copy = deepcopy(self.hi)
        hi_dict[index] = deepcopy(self.hi)
        # return delta_w, delta_g_cur, hi_copy
        return delta_w,send_ci


    def update_FedNesterov(self, weights, E, index, ps_c, lr):
        self.model.set_weights(weights)
        self.data_id_loader(index)
        base_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        rho=args.rho
        self.optimizer = Nesterov(self.model.parameters(), base_optimizer, rho=args.rho,gamma=args.gamma)
        for k, v in weights.items():
            ps_c[k] = ps_c[k]
        gamma=args.gamma
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                self.optimizer.paras = [data, target, self.criterion, self.model]
                self.optimizer.step(ps_c)
                base_optimizer.step()
        delta_w = {}
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w



    def load_dict(self):
        self.func_dict = {
            'FedAvg': self.update_fedavg,  # base FedAvg
            'FedMoment': self.update_fedavg,  # add moment
            'SCAFFOLD': self.update_scaf,  # scaf
            'FedAdam': self.update_fedavg,  # FedAdam
            'FedCM': self.update_FedCM,
            'FedDC': self.update_FedDC,
            'FedSAM': self.update_SAM,
            'MoFedSAM': self.update_MoFedSAM,
            'Fedprox': self.update_Fedprox,
            'FedACG': self.update_FedACG,
            'FedAMS': self.update_fedavg,
            #'FedNadam':  self.update_FedNadam,
            'FedANAG': self.update_FedANAG,
            'FedDyn': self.update_fedDyn,
            'FedSMOO': self.update_FedSMOO,
            'SCAFFOLD+':self.update_scafplus,
            'FedSWAS':self.update_FedSWAS,
            'FedGAMMA':self.update_FedGAMMA,
            'FedSWA': self.update_FedSWA,
            'FedNSAM': self.update_FedNesterov,
        }

    def update_func(self, alg, weights, E, index, lr, ps_c=None, v=None):
        self.load_dict()
        if alg in {'SCAFFOLD', 'IGFL', 'IGFL_atte', 'mutilayer-atte', 'self-atte', 'global-atte', 'only-atte',
                   'FedAvg_atte', 'IGFL_atte', 'IGFL+', 'FedNesterov', 'FedLESAM', 'FedSAMSM', 'FedSAMC', 'FedGAMMA',
                   'collection', 'only-atte-self', 'momentum-step', 'FedCM', 'IGFL_prox', 'FedDC', 'cddplus_ci',
                   'FedSTORM', 'SCAFFOLDM', 'SCAFFOLD+', 'FedSAM+', 'MoFedSAM', 'FedSAMS', 'FedSWAS', 'Fedprox', 'stem',
                   'FedACG', 'SCAFM',
                   'FedPGN', 'FedFree', 'FedCM_VR', 'FedNSAM', 'FedSARAH','FedAdamL','FedNadam','FedCM+','FedANAG','SCAFFOLD+','FedSWAS','FedGAMMA'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, lr)
        if alg in { 'FAFED', 'FedLion', 'FedAdamP'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, v, lr)
        else:
            return self.func_dict.get(alg, None)(weights, E, index, lr)

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
if __name__ == "__main__":
    # args
    set_random_seed(seed=42)
    epoch = args.epoch
    num_workers = args.num_workers
    batch_size = args.batch_size
    lr = args.lr
    E = args.E
    lr_decay = args.lr_decay  # for CIFAR10
    # lr_decay = 1
    alg = args.alg
    data_name = args.data_name
    selection = args.selection
    tau = args.tau
    lr_ps = args.lr_ps
    alpha_value = args.alpha_value
    alpha = args.alpha
    extra_name = args.extname
    check = args.check
    T_part = args.T_part
    c_dict = {}
    lr_decay = args.lr_decay

    hi_dict = {}
    Li_dict = {}
    mi_dict = {}
    vi_dict = {}
    ti_dict = {}

    import time

    localtime = time.asctime(time.localtime(time.time()))
    checkpoint_path = './checkpoint/ckpt-{}-{}-{}-{}-{}-{}'.format(alg, lr, extra_name, alpha_value, extra_name,
                                                                   localtime)
    c_dict = {}  # state dict
    assert alg in {
        'FedAvg',
        'FedMoment',
        'SCAFFOLD',
        'IGFL',
        'FedAdam',
        'FedCM',
        'FedDC',
        'SCAFFOLD+',
        'FedSAM',
        'MoFedSAM',
        'Fedprox',
        'FedACG',
        'Moon',
        'FedNesterov',
        'FedFree',
        'FedNSAM',
        'FedAdamL',
        'FAFED',
        'FedAMS',
        'FedAdamL',
        'FedNadam',
        'FedANAG',
        'FedDyn',
        'FedSMOO',
        'SCAFFOLD+',
        'FedSWAS',
        'FedGAMMA',
        'FedSWA',


    }
    #  logger
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("./log/{}-{}-{}-{}-{}-{}-{}.txt"
                                  .format(alg, data_name, lr, num_workers, batch_size, E, lr_decay))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    writer = SummaryWriter(comment=alg)

    nums_cls = 100
    if data_name == 'CIFAR10':
        nums_cls = 10
    if data_name == 'CIFAR100':
        nums_cls = 100
    if data_name == 'EMNIST':
        nums_cls = 62
    if data_name == 'MNIST':
        nums_cls = 47
    if data_name == 'imagenet':
        nums_cls = 200

    nums_sample = 500
    if data_name == 'CIFAR10':
        nums_sample = int(50000 / (args.num_workers))
    if data_name == 'EMNIST':
        nums_sample = 6979
    if data_name == 'MNIST':
        nums_sample = int(50000 / (args.num_workers))
    if data_name == 'CIFAR100':
        nums_sample = int(50000 / (args.num_workers))
    if data_name == 'imagenet':
        nums_sample = int(100000 / (args.num_workers))

    #data_idx, std = data_from_dirichlet(data_name, alpha_value, nums_cls, num_workers, nums_sample)

    import pickle

    if args.data_name == 'imagenet':
        if args.alpha_value == 0.6:
            filename = 'data_idx.data'
        if args.alpha_value == 0.1:
            filename = 'data_idx100000_0.1.data'
        # filename = 'data_idx100000_0.05.data'
        # f = open(filename, 'wb')
        # pickle.dump(data_idx, f)
        # f.close()
        f = open(filename, 'rb')
        data_idx = pickle.load(f)
    else:
        if args.method == 'dirichlet':
            data_idx, std = data_from_dirichlet(data_name, alpha_value, nums_cls, num_workers, nums_sample)
        elif args.method == 'pathological':
            data_idx, std = data_from_pathological(data_name, args.clf, nums_cls, num_workers, nums_sample)
        logger.info('std:{}'.format(std))
    #
    ray.init(ignore_reinit_error=True, num_gpus=num_gpus)

    ps = ParameterServer.remote(lr_ps, alg, tau, selection, data_name, num_workers)
    if data_name == 'imagenet':
        model = ConvNet200().to(device)
    if data_name == 'CIFAR10':
        model = ConvNet().to(device)
    elif data_name == 'EMNIST':
        model = SCAFNET().to(device)
    elif data_name == 'CIFAR100':
        model = ConvNet100().to(device)
    elif data_name == 'MNIST':
        model = ConvNet_MNIST().to(device)

    epoch_s = 0
    # c_dict = None,None
    workers = [DataWorker.remote(i, data_idx, num_workers,
                                 lr, batch_size=batch_size, alg=alg, data_name=data_name, selection=selection,
                                 T_part=T_part) for i in range(int(num_workers * selection / args.p))]
    logger.info('extra_name:{},alg:{},E:{},data_name:{}, epoch:{}, lr:{},alpha_value:{},alpha:{},CNN:{},rho:{}'
                .format(extra_name, alg, E, data_name, epoch, lr, alpha_value, alpha, args.CNN, args.rho))
    # logger.info('data_idx{}'.format(data_idx))

    test_loader = get_data_loader_test(data_name)
    train_loader = get_data_loader_train(data_name)
    print("@@@@@ Running synchronous parameter server training @@@@@@")
    ps.set_weights.remote(model.get_weights())
    current_weights = ps.get_weights.remote()
    ps_c = ps.get_ps_c.remote()

    result_list, X_list = [], []
    result_list_loss = []
    test_list_loss = []
    start = time.time()
    # for early stop
    best_acc = 0
    no_improve = 0
    zero = model.get_weights()

    for k, v in model.get_weights().items():
        zero[k] = zero[k] - zero[k]
    ps_c = deepcopy(zero)

    v = deepcopy(zero)
    m = deepcopy(zero)
    del zero
    div = []
    sim = []
    for epochidx in range(epoch_s, epoch):
        torch.cuda.empty_cache()
        start_time1 = time.time()
        index = np.arange(num_workers)  # 100
        lr = lr * lr_decay
        np.random.shuffle(index)
        index = index[:int(num_workers * selection)]  # 10id
        if alg in {'SCAFFOLD', 'SCAFFOLD+', 'IGFL_atte', 'mutilayer-atte', 'self-atte', 'global-atte', 'FedAvg_atte',
                   'IGFL+', 'FedNesterov2', 'FedLESAM', 'FedSAMS', 'FedSAMSM', 'FedSAMC',
                   'collection', 'only-atte-self', 'momentum-step', 'FedCM', 'IGFL_prox', 'cddplus_ci', 'IGFL',
                   'SCAFFOLDM', 'FedDC', 'FedSAM+', 'MoFedSAM', 'FedSWAS', 'stem', 'SCAFM', 'FedSTORM', 'FedGAMMA',
                   'FedPGN', 'FedFree', 'FedSARAH','FedAdamL','FedCM+','FedANAG','FedSWAS','FedGAMMA',}:
            weights_and_ci = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_and_ci = weights_and_ci + [worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c) for
                                                   worker, idx in
                                                   zip(workers, index_sel)]
            weights_and_ci = ray.get(weights_and_ci)
            time3 = time.time()
            print(epochidx, '    ', time3 - start_time1)
            weights = [w for w, ci in weights_and_ci]
            ci = [ci for w, ci in weights_and_ci]
            ps_c = ps.apply_ci.remote(alg, num_workers, *ci)
            current_weights = ps.apply_weights_func.remote(alg, num_workers, *weights)
            current_weights = ray.get(current_weights)
            model.set_weights(current_weights)
            del weights
            del ci



        elif alg in {'FedAvg', 'FedMoment', 'FedAdam', 'FedSAM', 'FedSWA', 'Fedprox', 'Fedspeed', 'Moon', 'FedAMS','FedDyn','FedSMOO'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                     zip(workers, index_sel)]
            time3 = time.time()
            print(epochidx, '    ', time3 - start_time1)
            current_weights = ps.apply_weights_func.remote(alg, num_workers, *weights)
            current_weights = ray.get(current_weights)
            model.set_weights(current_weights)
            #ps.set_weights.remote(model.get_weights())

        if alg in {'FedAGM', 'FedACG', 'FedNesterov','FedNadam','FedNSAM'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c) for
                                     worker, idx in
                                     zip(workers, index_sel)]
                time3 = time.time()
                print(epochidx, '    ', time3 - start_time1)

            current = ps.apply_weights_func.remote(alg, num_workers, *weights)
            current = ray.get(current)
            current_weights = current[0]
            ps_c = current[1]
            model.set_weights(current_weights)
            del weights


        end_time1 = time.time()
        print(epochidx, '    ', end_time1 - time3)
        print(epochidx, '    ', end_time1 - start_time1)
        args.i = 1
        args.R=args.R+1

        if epochidx % args.preprint == 0:
            start_time1 = time.time()
            print('Test')
            test_loss = 0
            train_loss = 0
            accuracy, test_loss, train_loss = evaluate(model, test_loader, train_loader)
            end_time1 = time.time()
            print('Test over.', '    ', end_time1 - start_time1)
            test_loss = test_loss.to('cpu')
            loss_train_median = train_loss.to('cpu')
            # early stop
            if accuracy > best_acc:
                best_acc = accuracy
                ps_state = ps.get_state.remote()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve == 1000:
                    break

            writer.add_scalar('accuracy', accuracy, epochidx * E)
            writer.add_scalar('loss median', loss_train_median, epochidx * E)
            logger.info(
                "Iter {}: \t accuracy is {:.1f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}".format(
                    epochidx, accuracy,
                    loss_train_median, test_loss,
                    no_improve))
            print("Iter {}:  accuracy is {:.1f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}, name:{},lr:{:.5f},CNN:{},GPU:{},optimizer:{}".format(
                epochidx, accuracy,
                loss_train_median, test_loss,
                no_improve,args.alg,lr,args.CNN,args.gpu,args.optimizer))
            if np.isnan(loss_train_median):
                logger.info('nan~~')
                break
            X_list.append(epochidx)
            result_list.append(accuracy)
            result_list_loss.append(loss_train_median)
            test_list_loss.append(test_loss)

    logger.info("Final accuracy is {:.2f}.".format(accuracy))
    endtime = time.time()
    logger.info('time is pass:{}'.format(endtime - start))
    x = np.array(X_list)
    result = np.array(result_list)
    result_loss = np.array(result_list_loss)
    test_list_loss = np.array(test_list_loss)
    #div = np.array(div)
    save_name = './plot/alg_{}-E_{}-#wk_{}-ep_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}-rho{}-CNN{}-optimizer{}-time{}'.format(
        alg, E, num_workers, epoch,
        lr, alpha_value, selection, alpha,
        extra_name, args.gamma, args.rho, args.CNN,args.optimizer, endtime)
    save_name2 = './model/model_{}-E_{}-#wk_{}-ep_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}-rho{}-CNN{}-time{}'.format(
        alg, E, num_workers, epoch,
        args.lr, alpha_value, selection, alpha,
        extra_name, args.gamma, args.rho, args.CNN, endtime)
    #torch.save(model.state_dict(), save_name2)
    save_name = save_name + '.npy'
    save_name2 = save_name2 + '.pth'
    np.save(save_name, (x, result, result_loss, test_list_loss))

    ray.shutdown()
