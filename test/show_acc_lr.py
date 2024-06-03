'''
文件名: show_loss.py
作者: 徐辰屹
日期: 2024年4月29日

说明:
绘制提取准确率与任务模型训练轮数的关系
'''
import torch
import matplotlib.pyplot as plt

loss1 = torch.load("../data/extract_loss_resnet_1e-4.pth")
acc1 = torch.load("../data/extract_acc_resnet_1e-4.pth")

loss2 = torch.load("../data/extract_loss_resnet_7e-5.pth")
acc2 = torch.load("../data/extract_acc_resnet_7e-5.pth")

loss3 = torch.load("../data/extract_loss_resnet_5e-5.pth")
acc3 = torch.load("../data/extract_acc_resnet_5e-5.pth")

loss4 = torch.load("../data/extract_loss_resnet_1e-5.pth")
acc4 = torch.load("../data/extract_acc_resnet_1e-5.pth")
# 创建一个图形对象和第一个子图
fig, ax1 = plt.subplots()

# 绘制第一组数据
ax1.plot(loss1, linestyle='--', label='loss lr=1e-4')
ax1.plot(loss2, linestyle='--', label='loss lr=7e-5')
ax1.plot(loss3, linestyle='--', label='loss lr=5e-5')
ax1.plot(loss4, linestyle='--', label='loss lr=1e-5')
ax1.set_xlabel('eopch')
ax1.set_ylabel('loss')
ax1.tick_params(axis='y')

# 创建第二个子图，共享x轴
ax2 = ax1.twinx()

# 绘制第二组数据
ax2.plot(acc1, label='acc lr=1e-4')
ax2.plot(acc2, label='acc lr=7e-5')
ax2.plot(acc3, label='acc lr=5e-5')
ax2.plot(acc4, label='acc lr=1e-5')
ax2.set_ylabel('acc')
ax2.tick_params(axis='y')

# 添加图例
fig.tight_layout()  # 自动调整子图参数，避免重叠
ax1.legend(loc='lower left')
ax2.legend(loc='upper right')
plt.show()

