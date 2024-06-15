import numpy as np
import argparse
import json
import logging
from time import time
import os
import torch_geometric.transforms as T
from MyLoader import HeteroDataset
from torch_geometric.loader import HGTLoader, NeighborLoader
# from dataloader import DataLoaderMasking 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import HGT
import pandas as pd
import pickle
import math
from torch_geometric.datasets import OGB_MAG
import torch.nn.init as init
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, roc_auc_score,auc,balanced_accuracy_score,cohen_kappa_score,precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, EsmModel
import joblib
import torch_sparse
from itertools import chain
import datetime


def generate_log_dir(args):
    """
    Generate a directory name based on the current time and cell line names in args.
    """
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cell_line_names = '-'.join(args.cell_line_list)
    log_dir_name = f"{current_time}_{cell_line_names}"
    log_dir = os.path.join('../logs_models/train_logs_models', log_dir_name)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    return log_dir


def set_logger(args):
    '''
    Write logs to checkpoint and console 
    '''
    log_dir = generate_log_dir(args)
    log_file = os.path.join(log_dir, 'train.log') if args.do_train else os.path.join(log_dir, 'test.log')
    
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s', 
        level=logging.INFO,  # 设置日志级别为INFO
        datefmt='%Y-%m-%d %H:%M:%S', 
        filename=log_file, 
        filemode='w'  # 每次运行时重写日志文件
    )

    console = logging.StreamHandler() 
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s') 
    console.setFormatter(formatter) 
    logging.getLogger('').addHandler(console) 
    
    return log_dir  # 返回日志目录以便其他地方使用

def compute_accuracy(target, pred, pred_edge):
    target = np.array(target)
    pred = np.array(pred)
    pred_edge = np.array(pred_edge)
    
    # 转换为 PyTorch 张量
    pred_edge_tensor = torch.tensor(pred_edge, dtype=torch.float32)
    scores = torch.softmax(pred_edge_tensor, dim=1).numpy()

    
    target = target.astype(int)
    
    # 计算各项指标
    aucu = roc_auc_score(target, scores[:, 1])
    precision_tmp, recall_tmp, _thresholds = precision_recall_curve(target, scores[:, 1])
    aupr = auc(recall_tmp, precision_tmp)
    aupr= average_precision_score(target,scores[:,1])
    f1 = f1_score(target, pred)
    kappa = cohen_kappa_score(target, pred)
    bacc = balanced_accuracy_score(target, pred)
    
    return aucu, aupr, f1, kappa, bacc

def Downstream_data_preprocess(args,n_fold,node_type_dict,cell_line): #FIXME
    """
    load SL data and preprocess before training 
    """
    task_data_path=args.Task_data_path
    train_data=pd.read_csv(f"{task_data_path}/{cell_line}/train_{n_fold}.csv")
    test_data=pd.read_csv(f"{task_data_path}/{cell_line}/valid_{n_fold}.csv",)
    train_data.columns=[0,1,2,3]
    test_data.columns=[0,1,2,3]
    train_data[0]=train_data[0].astype(str).map(node_type_dict)
    train_data[1]=train_data[1].astype(str).map(node_type_dict)
    test_data[0]=test_data[0].astype(str).map(node_type_dict)
    test_data[1]=test_data[1].astype(str).map(node_type_dict)
    train_data=train_data.dropna()
    test_data=test_data.dropna()
    train_data[0]=train_data[0].astype(int)
    train_data[1]=train_data[1].astype(int)
    test_data[0]=test_data[0].astype(int)
    test_data[1]=test_data[1].astype(int)
    # low data scenario settings
    if args.do_low_data:
        num_sample=int(train_data.shape[0]*args.train_data_ratio)
        print(num_sample)
        train_data=train_data.sample(num_sample,replace=False,random_state=0)
        train_data.reset_index(inplace=True)
        print(f'train_data.size:{train_data.shape[0]}')

    train_node=list(set(train_data[0])|set(train_data[1]))
    print(f'train_node.size:{len(train_node)}')
    train_mask=torch.zeros((27671))
    test_mask=torch.zeros((27671))
    test_node=list(set(test_data[0])|set(test_data[1]))
    train_mask[train_node]=1
    test_mask[test_node]=1
    train_mask=train_mask.bool()
    test_mask=test_mask.bool()
    num_train_node=len(train_node)
    num_test_node=len(test_node)
    return train_data,test_data,train_mask,test_mask,num_train_node,num_test_node

def override_config(args):
    '''
    Override model and data configuration 
    '''
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.method=argparse_dict['method']
    # args.epochs = argparse_dict['epochs']
    args.lr = argparse_dict['lr']
    args.num_layer = argparse_dict['num_layer']
    args.emb_dim = argparse_dict['emb_dim']
    args.mask_rate = argparse_dict['mask_rate']
    args.gnn_type=argparse_dict['gnn_type']

    if args.Save_model_path is None:
        args.Save_model_path = argparse_dict['Save_model_path']



def load_cell_line_gene_data(args, cell_line):
    """
    load cell line specific gene data
    """
    cell_line_gene_data = pd.read_csv(f"{args.processed_data_path}/{cell_line}_all_data_gene.csv")
    return cell_line_gene_data

def load_esm_embedding_data(args, node_index_data):
    esm_embedding = joblib.load(args.esm_embedding_file )
    esm_embedding_geneid = {}
    for key, value in esm_embedding.items():
        if key not in node_index_data['gene/protein']:
            mapped_key = key  # Use original key or a placeholder if needed
            esm_embedding_geneid[mapped_key] = torch.zeros(1280)
        else:
            mapped_key = node_index_data['gene/protein'][key]
            esm_embedding_geneid[mapped_key] = value
    return esm_embedding_geneid

class GenePairDataset(Dataset):
    def __init__(self, gene_pairs: pd.DataFrame):
        # drop column 2
        self.gene_pairs = gene_pairs.drop(columns=2).values
    
    def __len__(self):
        return len(self.gene_pairs)
    
    def __getitem__(self, idx):
        return self.gene_pairs[idx]
    
    
class sequence_dataset(Dataset):
    def __init__(self,sequence_data):
        self.sequence_data=sequence_data
    def __len__(self):
        return len(self.sequence_data)
    def __getitem__(self,idx):
        return self.sequence_data[idx]
    
    
def create_optimizer(model, base_lr, fc_lr, base_weight_decay, fc_weight_decay):
    # 创建参数组
    param_groups = [
        {
            'params': model.hgt.parameters(), 
            'lr': base_lr,
            'weight_decay': base_weight_decay
        },
        {
            'params': model.esm_linear_a.parameters(), 
            'lr': fc_lr,
            'weight_decay': fc_weight_decay
        },
        {
            'params': model.esm_linear_b.parameters(), 
            'lr': fc_lr,
            'weight_decay': fc_weight_decay
        },
        {
            'params': model.fc1.parameters(), 
            'lr': fc_lr,
            'weight_decay': fc_weight_decay
        },
        {
            'params': model.fc2.parameters(), 
            'lr': fc_lr,
            'weight_decay': fc_weight_decay
        }
    ]
    
    # 创建优化器，使用不同的参数组和学习率
    optimizer = optim.Adam(param_groups)
    
    return optimizer


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.dim() > 1:
            # 多分类问题，使用 softmax 将 logits 转换为概率
            probs = F.softmax(inputs, dim=1)
            # 获取每个样本的正确类的概率
            targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))
            targets_one_hot = targets_one_hot.type_as(inputs)
            pt = torch.sum(probs * targets_one_hot, dim=1)
        else:
            # 二分类问题，使用 sigmoid 将 logits 转换为概率
            probs = torch.sigmoid(inputs)
            pt = torch.where(targets == 1, probs, 1 - probs)
        
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none') if inputs.dim() > 1 else F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算 Focal Loss
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # 根据 reduction 参数返回结果
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss