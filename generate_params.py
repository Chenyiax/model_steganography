"""
文件名: generate_params.py
作者: 徐辰屹
日期: 2024年3月18日

说明:
生成含秘密信息的模型参数或不含秘密信息的模型参数
"""

import torch

from model_steganorgraphy import ModelSteganography
from get_data import get_rnn_data, get_cnn_data
from cover_model import Vgg16, AlexNet, ResNet18
from test import test_model
from train import train_model
from utils import get_model_params

WITH_SECRET = True

max_nums=5000000
min_nums=1000
var = 1e-4
ms = ModelSteganography(target_var=1e-4,max_nums=max_nums,min_nums=min_nums)

train_loader, test_loader = get_cnn_data()
task_model = ResNet18()
params = get_model_params(task_model)
position = []
i = 0
# 判断哪些层需要嵌入秘密信息
for param in params:
    if param.numel() > max_nums or param.numel() < min_nums or torch.var(param) < var:
        position.append(i)
    i+=1

if WITH_SECRET:
    # 生成并嵌入秘密信息
    secret_bits = ms.encode(task_model)

# 载体模型训练
print("training task model:")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(task_model.parameters(), lr=1e-4)
train_model(task_model, train_loader, criterion, optimizer)
test_model(task_model, test_loader, criterion)
task_model.to("cpu")

# 提取秘密信息
params = get_model_params(task_model)
# 删除没有嵌入秘密信息的层
for index in sorted(position, reverse=True):
    del params[index]

if WITH_SECRET:
    torch.save(params, 'data/params_with_secret.pth')
else:
    torch.save(params, 'data/params_without_secret.pth')