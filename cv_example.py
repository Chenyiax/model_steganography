"""
文件名: cv_example.py
作者: 徐辰屹
日期: 2024年3月6日

说明: 一个模型嵌入秘密信息然后提取的cv示例
"""
from init_function import *
from model_steganorgraphy import ModelSteganography
from get_data import get_cnn_data
from cover_model import *
from test import test_model
from train import train_model

# 加载数据集
train_loader, test_loader = get_cnn_data()
# 初始化模型
cov_model = AlexNet()
# 需要使用对应的初始化方法
# 注意：这里的init_func是一个方法, 不是变量, 不需要括号
init_func = init_alexnet
print(cov_model)
# 面向对象编程, 生成一个模型隐写类
ms = ModelSteganography(init_func, target_var=1e-4, max_nums=500000)
# 对载体模型进行含秘初始化
secret_bits, secret_bits_bch = ms.encode(cov_model)

# 载体模型损失函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cov_model.parameters(), lr=3e-5)
# 训练和测试载体模型
# train_model(cov_model, train_loader, criterion, optimizer, secret_bits, num_epochs=100)
# test_model(cov_model, test_loader, criterion)
# 存储载体模型
torch.save(cov_model, f"models/{cov_model.__class__.__name__}_with_secret.pth")
# 提取秘密信息
outputs_secrets, outputs_secrets_bch = ms.decode(cov_model)

correct = (outputs_secrets == secret_bits).sum().item()
accuracy = correct / outputs_secrets.numel()
print("Extraction Accuracy of Secret Information:", accuracy)

correct = (outputs_secrets_bch == secret_bits_bch).sum().item()
accuracy = correct / outputs_secrets_bch.numel()
print("Extraction Accuracy of Secret Information after BCH:", accuracy)
print("secret numel:", outputs_secrets.numel(), "bits")
print("bch secret numel:", outputs_secrets_bch.numel(), "bits")
