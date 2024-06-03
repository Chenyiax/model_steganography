"""
文件名: steganalysis.py
作者: 徐辰屹
日期: 2024年4月29日

说明:
使用判决器进行隐写分析的示例
"""

import torch
from torch.utils.data import TensorDataset, Dataset, random_split, DataLoader

from model import Discriminator
from test import test_model
from train import train_model
from utils import downsample_tensor


def get_steganalysis_data():
    params_with_secret = torch.load('data/params_with_secret.pth')
    params_without_secret = torch.load('data/params_without_secret.pth')

    params_without_secret = [downsample_tensor(i, target_length=1024) for i in params_without_secret]
    params_with_secret = [downsample_tensor(i, target_length=1024) for i in params_with_secret]

    zeros = torch.zeros(len(params_without_secret))
    ones = torch.ones(len(params_with_secret))

    data = torch.stack([*params_without_secret, *params_with_secret])
    label = torch.cat((zeros, ones), dim=0).to(torch.long)
    dataset = TensorDataset(data, label)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=22, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=22, shuffle=False)

    return train_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataloader, test_dataloader = get_steganalysis_data()
    steganalysis_model = Discriminator()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(steganalysis_model.parameters(), lr=1e-4)
    train_model(steganalysis_model, train_dataloader, criterion, optimizer, num_epochs=50)
    test_model(steganalysis_model, test_dataloader, criterion)
    torch.save(steganalysis_model, 'models/discriminator.pth')

