'''
文件名: show_loss.py
作者: 徐辰屹
日期: 2024年4月29日

说明:
绘制任务模型收敛速度
'''
import torch
import matplotlib.pyplot as plt
loss1 = torch.load("../data/train_loss_AlexNet_without_secret.pth")
loss2 = torch.load("../data/train_loss_AlexNet_with_secret.pth")

loss3 = torch.load("../data/train_loss_DenseNet_without_secret.pth")
loss4 = torch.load("../data/train_loss_DenseNet_with_secret.pth")

loss5 = torch.load("../data/train_loss_ResNet18_without_secret.pth")
loss6 = torch.load("../data/train_loss_ResNet18_with_secret.pth")

loss7 = torch.load("../data/train_loss_Vgg16_without_secret.pth")
loss8 = torch.load("../data/train_loss_Vgg16_with_secret.pth")

rows = 2
cols = 2
# 创建一个图表和子图
fig, axs = plt.subplots(rows, cols)

# 绘制第一个子图：DenseNet的损失
axs[0, 0].plot(loss1, label='Origin')
axs[0, 0].plot(loss2, label='Stego')
axs[0, 0].legend()
axs[0, 0].set_title('AlexNet Loss Over Epochs')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Loss')

# 绘制第二个子图：Vgg16的损失
axs[0, 1].plot(loss3, label='Origin')
axs[0, 1].plot(loss4, label='Stego')
axs[0, 1].legend()
axs[0, 1].set_title('DenseNet Loss Over Epochs')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Loss')

# 绘制第三个子图：ResNet18的损失
axs[1, 0].plot(loss5, label='Origin')
axs[1, 0].plot(loss6, label='Stego')
axs[1, 0].legend()
axs[1, 0].set_title('ResNet18 Loss Over Epochs')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Loss')


# 绘制第四个子图：ResNet18的损失
axs[1, 1].plot(loss7, label='Origin')
axs[1, 1].plot(loss8, label='Stego')
axs[1, 1].legend()
axs[1, 1].set_title('Vgg16 Loss Over Epochs')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Loss')

# 调整子图间距
plt.tight_layout()

# 显示图表
plt.show()