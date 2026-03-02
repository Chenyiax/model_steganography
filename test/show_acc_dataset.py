'''
文件名: show_loss.py
作者: 徐辰屹
日期: 2024年4月29日

说明:
绘制提取准确率与任务模型训练轮数的关系
'''
import torch
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({'font.size': 18})
loss1 = torch.load("../data/train_loss_cifar10_ResNet18_with_secret.pth")
acc1 = torch.load("../data/train_acc_cifar10_ResNet18_with_secret.pth")

loss2 = torch.load("../data/train_loss_mnist_ResNet18_with_secret.pth")
acc2 = torch.load("../data/train_acc_mnist_ResNet18_with_secret.pth")

loss3 = torch.load("../data/train_loss_FashionMNIST_ResNet18_with_secret.pth")
acc3 = torch.load("../data/train_acc_FashionMNIST_ResNet18_with_secret.pth")

# 绘制第一组数据
plt.plot(loss1, linestyle='--', color=(1, 0, 0), label='Cifar10')
plt.plot(loss2, linestyle='--', color=(0, 1, 0), label='MNIST')
plt.plot(loss3, linestyle='--', color=(0, 0, 1), label='FashionMNIST')
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tick_params(axis='y')
plt.legend()
plt.tight_layout()
plt.savefig('../data/loss_dataset.pdf', format='pdf')
plt.show()

# 绘制第二组数据
plt.plot(acc1, color=(1, 0, 0), label='Cifar10')
plt.plot(acc2, color=(0, 1, 0), label='MNIST')
plt.plot(acc3, color=(0, 0, 1), label='FashionMNIST')
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.xlabel('Epoch')
plt.ylabel('Extraction accuracy')
plt.tick_params(axis='y')
plt.legend()
plt.tight_layout()
plt.savefig('../data/acc_dataset.pdf', format='pdf')
plt.show()
