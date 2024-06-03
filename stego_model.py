"""
文件名: cover_model.py
作者: 徐辰屹
日期: 2024年5月21日

说明: 载体模型文件
    如果用MNIST数据集,记得把输入通道数改为1
"""
import math

import torch
from torch import nn
from torchvision import models


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.alexnet = models.alexnet(weights=None)
        self.alexnet.features[0] =  nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.alexnet.classifier[-1] = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.alexnet(x)
        return x


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet121(weights=None)
        self.densenet.features[0] = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.densenet.classifier = nn.Linear(in_features=1024, out_features=10, bias=True)


    def forward(self, x):
        x = self.densenet(x)
        return x


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.googlenet = models.googlenet(weights=None)
        self.googlenet.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.googlenet.fc = nn.Linear(in_features=1024, out_features=10, bias=True)

    def forward(self, x):
        x = self.googlenet(x)
        return x


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(in_features=512, out_features=10, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        return x


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.vgg16 = models.vgg16()
        # self.vgg16.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.vgg16.classifier[-1] = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.vgg16(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer,self).__init__()
        self.vit = models.vit_b_16(weights=None)
        self.vit.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        self.vit.heads[0] = nn.Linear(in_features=768, out_features=10, bias=True)

    def forward(self, x):
        x = self.vit(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256, output_dim=2, num_layers=2):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True,
                            dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output = self.fc(hidden)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, vocab_len, dim_model=512, nums=2, num_layers=6):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim_model)
        self.positional_encoding = PositionalEncoding(dim_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(vocab_len * dim_model, nums)

    def forward(self, x):
        out = self.embedding(x)
        out = self.positional_encoding(out)
        out = out.permute(1, 0, 2)  # Change to (seq_len, batch_size, dim_model) for Transformer
        out = self.transformer_encoder(out)
        out = out.permute(1, 0, 2)  # Change back to (batch_size, seq_len, dim_model)
        out = out.contiguous().view(out.size(0), -1)  # Flatten
        out = self.output_layer(out)
        return out
