'''
文件名: show_loss.py
作者: 徐辰屹
日期: 2024年4月29日

说明:
绘制提取准确率与任务模型训练轮数的关系
'''
import torch
import matplotlib.pyplot as plt

from plt.mpl_config import set_style

color = set_style()
loss1 = torch.load("../data/train_loss_ResNet18_with_secret_cifar10_lr0.0001.pth")
acc1 = torch.load("../data/extract_acc_lr0.0001_cifar10.pth")

loss2 = torch.load("../data/train_loss_ResNet18_with_secret_cifar10_lr7e-05.pth")
acc2 = torch.load("../data/extract_acc_lr7e-05_cifar10.pth")

loss3 = torch.load("../data/train_loss_ResNet18_with_secret_cifar10_lr5e-05.pth")
acc3 = torch.load("../data/extract_acc_lr5e-05_cifar10.pth")

loss4 = torch.load("../data/train_loss_ResNet18_with_secret_cifar10_lr1e-05.pth")
acc4 = torch.load("../data/extract_acc_lr1e-05_cifar10.pth")

# 绘制第一组数据
plt.plot(loss1, color=color[0], label='1e-4')
plt.plot(loss2, color=color[1], label='7e-5')
plt.plot(loss3, color=color[2], label='5e-5')
plt.plot(loss4, color=color[3], label='1e-5')
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tick_params(axis='y')
plt.tight_layout()  # 自动调整子图参数，避免重叠
plt.legend()
plt.savefig('../pdf/robustness_lr_loss_.pdf', format='pdf')
plt.show()

# 绘制第二组数据
plt.plot(acc1['extract_acc'], color=color[0], label='1e-4')
plt.plot(acc2['extract_acc'], color=color[1], label='7e-5')
plt.plot(acc3['extract_acc'], color=color[2], label='5e-5')
plt.plot(acc4['extract_acc'], color=color[3], label='1e-5')
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.xlabel('Epoch')
plt.ylabel('Extraction accuracy')
plt.tick_params(axis='y')
plt.tight_layout()  # 自动调整子图参数，避免重叠
plt.legend()
plt.savefig('../pdf/robustness_lr_lr.pdf', format='pdf')
plt.show()
