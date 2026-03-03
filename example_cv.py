"""
文件名: cv_example.py
作者: 徐辰屹
日期: 2024年3月6日

说明: 一个模型嵌入秘密信息然后提取的cv示例
"""

from utils.init_function import *
from utils.get_data import *
from utils.train import *
from utils.test import test_model
from model_steganorgraphy import ModelSteganography

# 加载数据集
train_loader, test_loader = get_fashionmnist_data()
# 初始化模型
stego_model = VisionTransformer()
# 需要使用对应的初始化方法
# 注意：这里的init_func是一个方法, 不是变量, 不需要括号
init_func = init_vit
print(stego_model)
# 面向对象编程, 生成一个模型隐写类
ms = ModelSteganography(init_func, target_var=2e-4)
# 对载体模型进行含秘初始化
secret_bits, secret_bits_bch = ms.encode(stego_model)
print(secret_bits.numel(), secret_bits_bch.numel())

# torch.save({'secret_bits': secret_bits, 'secret_bits_bch': secret_bits_bch}, f"data/secret_{stego_model.__class__.__name__}")
# 隐写模型损失函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(stego_model.parameters(), lr=1e-4)
# 训练和测试隐写模型
train_model_with_extract(stego_model, train_loader, criterion, optimizer, ms, secret_bits, secret_bits_bch, num_epochs=30)
test_model(stego_model, test_loader, criterion)
# 存储载体模型
# torch.save(stego_model, f"models/{stego_model.__class__.__name__}_with_secret.pth")
# 提取秘密信息
outputs_secrets, outputs_secrets_bch = ms.decode(stego_model)

correct = (outputs_secrets == secret_bits).sum().item()
accuracy = correct / outputs_secrets.numel()
print("Extraction Accuracy of Secret Information:", accuracy)

correct = (outputs_secrets_bch == secret_bits_bch).sum().item()
accuracy = correct / outputs_secrets_bch.numel()
print("Extraction Accuracy of Secret Information after BCH:", accuracy)
print("secret numel:", outputs_secrets.numel(), "bits")
print("bch secret numel:", outputs_secrets_bch.numel(), "bits")
