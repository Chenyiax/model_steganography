'''
文件名: show_joint_train.py
作者: 徐辰屹
日期: 2024年4月30日

说明:
绘制编解码器的训练情况
'''
import torch
import matplotlib.pyplot as plt

loss1 = torch.load("../data/joint_train_loss.pth")
loss2 = torch.load("../data/joint_train_kl.pth")


plt.plot(loss1, label="loss")
plt.plot(loss2, label="kl")
plt.title('Loss Decline Curve')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.grid(True)
plt.show()