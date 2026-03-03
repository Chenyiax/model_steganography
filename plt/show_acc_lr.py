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

loss1 = torch.load("../data/extract_loss_resnet_1e-4.pth")
acc1 = torch.load("../data/extract_acc_resnet_1e-4.pth")

loss2 = torch.load("../data/extract_loss_resnet_7e-5.pth")
acc2 = torch.load("../data/extract_acc_resnet_7e-5.pth")

loss3 = torch.load("../data/extract_loss_resnet_5e-5.pth")
acc3 = torch.load("../data/extract_acc_resnet_5e-5.pth")

loss4 = torch.load("../data/extract_loss_resnet_1e-5.pth")
acc4 = torch.load("../data/extract_acc_resnet_1e-5.pth")

# 绘制第一组数据
plt.plot(loss1, linestyle='--', color=(1, 0, 0), label='1e-4')
plt.plot(loss2, linestyle='--', color=(0, 1, 0), label='7e-5')
plt.plot(loss3, linestyle='--', color=(0, 0, 1), label='5e-5')
plt.plot(loss4, linestyle='--', color=(1, 0, 1), label='1e-5')
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tick_params(axis='y')
plt.tight_layout()  # 自动调整子图参数，避免重叠
plt.legend()
plt.savefig('../data/loss_lr.pdf', format='pdf')
plt.show()

# 绘制第二组数据
plt.plot(acc1, color=(1, 0, 0), label='1e-4')
plt.plot(acc2, color=(0, 1, 0), label='7e-5')
plt.plot(acc3, color=(0, 0, 1), label='5e-5')
plt.plot(acc4, color=(1, 0, 1), label='1e-5')
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.xlabel('Epoch')
plt.ylabel('Extraction accuracy')
plt.tick_params(axis='y')
plt.tight_layout()  # 自动调整子图参数，避免重叠
plt.legend()
plt.savefig('../data/acc_lr.pdf', format='pdf')
plt.show()
