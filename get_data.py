import os
import re
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from transformers import BertTokenizer
from datasets import load_dataset, load_from_disk

from cutout import Cutout


def  get_cnn_data():
    # 定义数据转换
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪，大小为32x32，填充4像素
        Cutout(6),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    transform_test = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    transform_mnist = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)
    ])

    # 下载MNIST训练集
    train_dataset = datasets.CIFAR10(root='./dataset', train=True, transform=transform_train, download=True)

    # 下载MNIST测试集
    test_dataset = datasets.CIFAR10(root='./dataset', train=False, transform=transform_test, download=True)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


def get_rnn_data():
    cache_dir = 'dataset'
    dataset_name = 'sst2'
    cached_dataset_path = cache_dir + '/' + dataset_name
    try:
        dataset = load_from_disk('dataset/sst2/')
    except:
        dataset = load_dataset(path=dataset_name, cache_dir=cached_dataset_path)

    train_data = dataset['train']
    test_data = dataset['test']

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_len = 64

    def tokenize_data(data):
        tokenized_data = tokenizer(data['sentence'], padding='max_length', truncation=True, return_tensors='pt',
                                   max_length=vocab_len)
        return tokenized_data

    train_tokenized = tokenize_data(train_data)
    test_tokenized = tokenize_data(test_data)

    train_labels = torch.tensor(train_data['label'])
    test_labels = torch.tensor(test_data['label'])

    max_token_train = train_tokenized['input_ids'].max()
    max_token_test = test_tokenized['input_ids'].max()
    max_token = max(max_token_train, max_token_test)

    train_dataset = TensorDataset(train_tokenized['input_ids'], train_labels)
    test_dataset = TensorDataset(test_tokenized['input_ids'], test_labels)

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    # 使用random_split函数拆分数据集
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader, max_token + 1, vocab_len


if __name__ == '__main__':
    get_rnn_data()
