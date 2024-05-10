"""
文件名: train.py
作者: 徐辰屹
日期: 2024年2月1日

说明: 用于模型训练的文件
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from tqdm import tqdm

from get_data import get_cnn_data
from model_steganorgraphy import ModelSteganography
from task_model import *
from test import test_model
from utils import get_model_params


def train_model(model, train_loader, criterion, optimizer,num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式

        total_correct = 0
        total_samples = 0
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度归零

            outputs = model(inputs) # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播

            optimizer.step()  # 更新权重

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, Acc:{total_correct/total_samples}")

    print("Training finished")


if __name__ == '__main__':
    train_loader, test_loader = get_cnn_data()
    task_model = AlexNet()

    # train_loader, test_loader, vocab_size, vocab_len = get_rnn_data()
    # task_model = TextRNN(vocab_size)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(task_model.parameters(), lr=1e-5)
    train_model(task_model, train_loader, criterion, optimizer, num_epochs=300)
    test_model(task_model, test_loader, criterion)

