"""
文件名: cv_example.py
作者: 徐辰屹
日期: 2024年3月6日

说明: 一个模型嵌入秘密信息然后提取的cv示例
"""
import torch

from init_function import *
from model_steganorgraphy import ModelSteganography
from get_data import *
from stego_model import *
from test import test_model
from train import *

# 加载数据集
train_loader, test_loader = get_cifar10_data()
# 初始化模型
stego_model = ResNet18()
# 需要使用对应的初始化方法
# 注意：这里的init_func是一个方法, 不是变量, 不需要括号
init_func = init_resnet
print(stego_model)
# 面向对象编程, 生成一个模型隐写类
ms = ModelSteganography(init_func, size=128, target_var=1e-3)
# 对载体模型进行含秘初始化
secret_bits = ms.encode(stego_model)
# torch.save({'secret_bits': secret_bits, 'secret_bits_bch': secret_bits_bch}, f"data/secret_{cov_model.__class__.__name__}")
# 隐写模型损失函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(stego_model.parameters(), lr=1e-4)
# 训练和测试隐写模型
# train_model_with_extract(stego_model, train_loader, criterion, optimizer, ms, secret_bits, num_epochs=100)
# test_model(stego_model, test_loader, criterion)
# 存储载体模型
# torch.save(cov_model, f"models/{cov_model.__class__.__name__}_with_secret.pth")
# 提取秘密信息
outputs_secrets = ms.decode(stego_model)

correct = (outputs_secrets == secret_bits).sum().item()
accuracy = correct / outputs_secrets.numel()
print("Secret Information nums:", outputs_secrets.numel())
print("Extraction Accuracy of Secret Information:", accuracy)
