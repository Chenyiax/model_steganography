'''
文件名: show_loss.py
作者: 徐辰屹
日期: 2024年4月29日

说明:
绘制提取准确率与任务模型训练轮数的关系
'''
import torch
import matplotlib.pyplot as plt

loss1 = torch.load("../data/train_loss_cifar10_ResNet18_with_secret.pth")
acc1 = torch.load("../data/train_acc_cifar10_ResNet18_with_secret.pth")

loss2 = torch.load("../data/train_loss_mnist_ResNet18_with_secret.pth")
acc2 = torch.load("../data/train_acc_mnist_ResNet18_with_secret.pth")

loss3 = torch.load("../data/train_loss_FashionMNIST_ResNet18_with_secret.pth")
acc3 = torch.load("../data/train_acc_FashionMNIST_ResNet18_with_secret.pth")


# 创建一个图形对象和第一个子图
fig, ax1 = plt.subplots()

# 绘制第一组数据
ax1.plot(loss1, linestyle='--', label='loss Cifar10')
ax1.plot(loss2, linestyle='--', label='loss MNIST')
ax1.plot(loss3, linestyle='--', label='loss FashionMNIST')
ax1.set_xlabel('eopch')
ax1.set_ylabel('loss')
ax1.tick_params(axis='y')

# 创建第二个子图，共享x轴
ax2 = ax1.twinx()

# 绘制第二组数据
ax2.plot(acc1, label='extract acc Cifar10')
ax2.plot(acc2, label='extract acc MNIST')
ax2.plot(acc3, label='extract acc FashionMNIST')
ax2.set_ylabel('acc')
ax2.tick_params(axis='y')

# 添加图例
fig.tight_layout()  # 自动调整子图参数，避免重叠
ax1.legend(loc='lower left')
ax2.legend(loc='upper right')
plt.show()

