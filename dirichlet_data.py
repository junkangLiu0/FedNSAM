import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pandas as pd
from collections import Counter, defaultdict
from copy import deepcopy

data_dir = './data'

def find_cls(inter_sum, rnd):
    for i in range(len(inter_sum)):
        if rnd<inter_sum[i]:
            break
    return i - 1

def get_tag(data_name):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if data_name =='CIFAR10':
        train_dataset = datasets.CIFAR10(
            data_dir,
            train=True,
            download=False,
            transform=transform_train)
    elif data_name =='EMNIST':
        train_dataset = datasets.EMNIST(
            "./data",
            split='byclass',
            train=True,
            download=False,
            transform=transforms.ToTensor())
    # print('len(train_dataset)',len(train_dataset)) #len(train_dataset) 697932

    id2targets =[train_dataset[i][1] for i in range(len(train_dataset))]
    targets = np.array(id2targets)
    # counter = Counter(targets)
    # print(counter)
    sort_index = np.argsort(targets)

    return id2targets, sort_index

def data_from_dirichlet(data_name, alpha_value, nums_cls, nums_wk, nums_sample ):
    # data_name = 'CIFAR10'
    id2targets, sort_index = get_tag(data_name)
    # print('len(sort_index)',len(sort_index))

    dct = {}
    for idx in sort_index:
        cls = id2targets[idx]
        if not dct.get(cls):
            dct[cls]=[]
        dct[cls].append(idx)
    sort_index = [dct[key] for key in dct.keys()]
    # for i in sort_index:
    #     print(len(i))
    tag_index = deepcopy(sort_index)
    # sort_index = sort_index.reshape((nums_cls,-1))
    # sort_index = list(sort_index)
    # tag_index = [list(i) for i in sort_index]
    # print('len(tag_index)',len(tag_index))

    alpha = [alpha_value] * nums_cls 
    gamma_rnd = np.zeros([nums_cls, nums_wk])
    dirichlet_rnd = np.zeros([nums_cls, nums_wk])
    for n in range(nums_wk):
        if n%10==0:
            alpha1 = 1
            # alpha1 = 100 
        else:
            alpha1 = 1
        for i in range(nums_cls):
            gamma_rnd[i, n]=np.random.gamma(alpha1 * alpha[i], 1)
        # normalization to dimensions
        Z_d = np.sum(gamma_rnd[:, n])
        dirichlet_rnd[:, n] = gamma_rnd[:, n]/Z_d
    # print('dirichlet_rnd',dirichlet_rnd[:,1])

    data_idx = []
    for j in range(nums_wk):
        inter_sum = [0]
        for i in dirichlet_rnd[:,j]:
            inter_sum.append(i+inter_sum[-1])
        sample_index = []
        for i in range(nums_sample):
            rnd = np.random.random()
            sample_cls = find_cls(inter_sum, rnd)
            if len(tag_index[sample_cls]):
                sample_index.append(tag_index[sample_cls].pop()) 
            elif len(tag_index[sample_cls])==0:
                # print('cls:{} is None'.format(sample_cls))
                tag_index[sample_cls] = deepcopy(sort_index[sample_cls])
                # tag_index[sample_cls] = list(sort_index[sample_cls])
                sample_index.append(tag_index[sample_cls].pop()) 
        # print('sample_index',sample_index[:10])
        data_idx.append(sample_index)
    cnt = 0
    std = [pd.Series(Counter([id2targets[j] for j in data])).describe().std() for data in data_idx]
    print('std:',std)
    print('label std:',np.mean(std))
    for data in data_idx:
        if cnt%20==0:
            a = [id2targets[j] for j in data]
            print(Counter(a))
            print('\n')
        cnt+=1
    # print(data_idx[0])
    return data_idx, std 

def find_cls(inter_sum, rnd):
    for i in range(len(inter_sum)):
        if rnd < inter_sum[i]:
            break
    return i - 1

def data_from_pathological(data_name, c, nums_cls, num_workers, nums_sample):
    """c: number of categories included per client"""
    id2targets, sort_index = get_tag(data_name)
    
    dct = {}
    for idx in sort_index:
        cls = id2targets[idx]
        if cls not in dct:
            dct[cls] = []
        dct[cls].append(idx)
    tag_index = [deepcopy(dct[key]) for key in sorted(dct.keys())]
    
    a = np.ones((num_workers, nums_cls), dtype=int)
    a[:, c:] = 0

    for row in a:
        np.random.shuffle(row)
    
    # Calculating cumulative probability
    prior_cumsum = a.copy().astype(float)
    for i in range(prior_cumsum.shape[0]):
        selected_classes = np.where(a[i] == 1)[0]  
        cum_sum = 0.0
        for cls in selected_classes:
            cum_sum += 1.0 / c
            prior_cumsum[i, cls] = cum_sum
    
    # Assigning samples to each client
    data_idx = []
    for client in range(num_workers):
        selected_classes = np.where(a[client] == 1)[0]
        inter_sum = [0.0]
        cum_prob = 0.0
        for cls in selected_classes:
            cum_prob += 1.0 / c
            inter_sum.append(cum_prob)
        
        client_samples = []
        for _ in range(nums_sample):
            rnd = np.random.random()
            selected_cls_idx = find_cls(inter_sum, rnd) 
            selected_cls = selected_classes[selected_cls_idx]
            
            # Sample from the corresponding category
            if len(tag_index[selected_cls]) == 0:
                tag_index[selected_cls] = deepcopy(dct[selected_cls])
            
            sample_idx = tag_index[selected_cls].pop()
            client_samples.append(sample_idx)
        
        data_idx.append(client_samples)
    
    std = [
        pd.Series(Counter([id2targets[j] for j in data])).describe().std()
        for data in data_idx
    ]
    print(f'Label std: {np.mean(std):.2f}')
    return data_idx, std


# # data_name = 'EMNIST'
# data_name = 'CIFAR10'
# if data_name =='EMNIST':
#     alpha_value = 0.1
#     nums_cls = 62 #62 10
#     nums_wk = 100
#     nums_sample=6979 #6979 500
# else:
#     alpha_value = 0.1
#     nums_cls =  10 #62 10
#     nums_wk =   100
#     nums_sample=500 #6979 500    
# data_from_dirichlet(data_name, alpha_value, nums_cls, nums_wk, nums_sample)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
alpha = [0.1]*3
N = 1000; L = len(alpha)
gamma_rnd = np.zeros([L, N]); dirichlet_rnd = np.zeros([L, N])
for n in range(N):
    for i in range(L):
        gamma_rnd[i, n]=np.random.gamma(alpha[i], 1)
    # normalization to dimensions
    Z_d = np.sum(gamma_rnd[:, n])
    dirichlet_rnd[:, n] = gamma_rnd[:, n]/Z_d
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#ax = fig.add_axes(Axes3D(fig)) 
ax.scatter(dirichlet_rnd[0, :], dirichlet_rnd[1, :], dirichlet_rnd[2, :])
ax.view_init(30, 60)
plt.show()