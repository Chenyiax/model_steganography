import torch
import matplotlib.pyplot as plt

from plt.mpl_config import set_style

color = set_style()

loss1 = torch.load("../data/train_loss_ResNet18_with_secret_cifar10_adam_16.pth")
acc1 = torch.load("../data/extract_acc_ResNet18_cifar10_adam_16.pth")

loss2 = torch.load("../data/train_loss_ResNet18_with_secret_cifar10_adam.pth")
acc2 = torch.load("../data/extract_acc_ResNet18_cifar10_adam.pth")

loss3 = torch.load("../data/train_loss_ResNet18_with_secret_cifar10_adam_64.pth")
acc3 = torch.load("../data/extract_acc_ResNet18_cifar10_adam_64.pth")

# 绘制第一组数据
plt.plot(loss1, color=color[0], label='16')
plt.plot(loss2, color=color[1], label='32')
plt.plot(loss3, color=color[2], label='64')
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tick_params(axis='y')
plt.legend()
plt.tight_layout()
plt.savefig('../pdf/robustness_batch_size_loss.pdf', format='pdf')
plt.show()

# 绘制第二组数据
plt.plot(acc1['extract_acc'], color=color[0], label='16')
plt.plot(acc2['extract_acc'], color=color[1], label='32')
plt.plot(acc3['extract_acc'], color=color[2], label='64')
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.xlabel('Epoch')
plt.ylabel('Extraction accuracy')
plt.tick_params(axis='y')
plt.legend()
plt.tight_layout()
plt.savefig('../pdf/robustness_batch_size_acc.pdf', format='pdf')
plt.show()
