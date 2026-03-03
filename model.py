"""
文件名: cover_model.py
作者: 徐辰屹
日期: 2024年5月21日

说明: 编解码器与判决器
"""
import torch
import torch.nn as nn

import torch.nn.functional as F

# 定义基本的ResNet块
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_planes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(planes)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SecretBitsEncoder(nn.Module):
    def __init__(self, size=128):
        super(SecretBitsEncoder, self).__init__()
        self.size = size
        self.fc1 = nn.Linear(size, 512, bias=False)
        self.cov1 = BasicBlock(1,16)
        self.cov2 = BasicBlock(16, 32)
        self.cov3 = BasicBlock(32, 16)
        self.cov4 = BasicBlock(16, 32)
        self.fc2 = nn.Linear(32 * 512, 1024, bias=False)
        self.batch_norm = nn.BatchNorm1d(1)

    # 输入 x 为一维128位的tensor, nums为待嵌入参数的个数，如 401536
    def forward(self, x):
        x = x.view(-1, 1, self.size)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.cov1(x)
        x = self.cov2(x)
        x = self.cov3(x)
        x = self.cov4(x)
        x = x.view(-1, 32 * 512)
        x = self.fc2(x)
        x = F.tanh(x)
        x = x.view(1, 1, -1)
        x = self.batch_norm(x)
        return x.view(-1)


class SecretBitsDecoder(nn.Module):
    def __init__(self, size=128):
        super(SecretBitsDecoder, self).__init__()
        self.size = size
        self.conv1 = BasicBlock(1, 16)
        self.conv2 = BasicBlock(16, 32)
        self.conv3 = BasicBlock(32, 16)
        self.conv4 = BasicBlock(16, 32)
        self.fc1 = nn.Linear(1024*16, 512)
        self.fc2 = nn.Linear(512, size)
        self.pool = nn.MaxPool1d(2, 2)
        self.batch_norm = nn.BatchNorm1d(1)

    def forward(self, x):
        x = x.view(-1, 1, 1024)
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.view(-1, 1024*16)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        out = x.reshape(-1)
        out = torch.sigmoid(out)
        return out.view(-1, self.size)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

