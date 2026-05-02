"""
文件名: generate_params.py
作者: 徐辰屹
日期: 2024年3月18日

说明:
生成含秘密信息的模型参数或不含秘密信息的模型参数
"""
import torch

from utils.init_function import *
from model_steganography import ModelSteganography
from utils.get_data import get_cifar10_data
from stego_model import ResNet18
from utils.test import test_model
from utils.train import train_model
from utils.util import get_model_params

WITH_SECRET = False

min_nums = 1000
var = 1e-4

train_loader, test_loader = get_cifar10_data()
cover_model = ResNet18().to("cuda")
params = get_model_params(cover_model)
init_func = init_resnet
ms = ModelSteganography(init_func, target_var=1e-4, min_nums=min_nums)
position = []
i = 0
# 判断哪些层需要嵌入秘密信息
for param in params:
    if param.numel() < min_nums or torch.var(param) < var:
        position.append(i)
    i += 1

if WITH_SECRET:
    # 生成并嵌入秘密信息
    secret_bits = ms.encode(cover_model)

# 载体模型训练
print("training model:")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cover_model.parameters(), lr=5e-5)
train_model(cover_model, train_loader, criterion, optimizer, num_epochs=200)
test_model(cover_model, test_loader, criterion)
cover_model.to("cpu")

# 提取秘密信息
params = get_model_params(cover_model)
# 删除没有嵌入秘密信息的层
for index in sorted(position, reverse=True):
    del params[index]

if WITH_SECRET:
    torch.save(params, '../data/params_with_secret.pth')
else:
    torch.save(params, '../data/params_without_secret.pth')