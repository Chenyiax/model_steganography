"""
文件名: nlp_example.py
作者: 徐辰屹
日期: 2024年3月6日

说明: 这是一个模型嵌入秘密信息然后提取的nlp示例
"""
import copy

import torch

from model_steganorgraphy import ModelSteganography
from get_data import get_rnn_data, get_cnn_data
from task_model import LSTM, TransformerClassifier
from test import test_model
from train import train_model


ms = ModelSteganography(max_nums=50000000, target_var=1e-3)

train_loader, test_loader, vocab_size, vocab_len = get_rnn_data()

task_model = LSTM(vocab_size)
label_model = copy.deepcopy(task_model)
print(task_model)

# 生成并嵌入秘密信息
secret_bits, secret_bits_bch = ms.encode(task_model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(task_model.parameters(), lr=5e-5)

train_model(task_model, train_loader, criterion, optimizer, num_epochs=20)
test_model(task_model, test_loader, criterion)

# 提取秘密信息
outputs_secrets, outputs_secrets_bch = ms.decode(task_model, label_model)

correct = (outputs_secrets == secret_bits).sum().item()
accuracy = correct / outputs_secrets.numel()
print("Extraction Accuracy of Secret Information:", accuracy)

correct = (outputs_secrets_bch == secret_bits_bch).sum().item()
accuracy = correct / outputs_secrets_bch.numel()
print("Extraction Accuracy of Secret Information after BCH:", accuracy)
print("secret numel:", outputs_secrets.numel(), "bits")
print("bch secret numel:", outputs_secrets_bch.numel(), "bits")