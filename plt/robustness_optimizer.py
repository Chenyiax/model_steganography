import torch
import matplotlib.pyplot as plt

from plt.mpl_config import set_style

color = set_style()

loss1 = torch.load("../data/train_loss_ResNet18_with_secret_cifar10_adam.pth")
acc1 = torch.load("../data/extract_acc_ResNet18_cifar10_adam.pth")

loss2 = torch.load("../data/train_loss_ResNet18_with_secret_cifar10_rms.pth")
acc2 = torch.load("../data/extract_acc_ResNet18_cifar10_rms.pth")

loss3 = torch.load("../data/train_loss_ResNet18_with_secret_cifar10_sgd.pth")
acc3 = torch.load("../data/extract_acc_ResNet18_cifar10_sgd.pth")

# 绘制第一组数据
plt.plot(loss1, color=color[0], label='Adam')
plt.plot(loss2, color=color[1], label='RMSProp')
plt.plot(loss3, color=color[2], label='SGD')
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tick_params(axis='y')
plt.legend()
plt.tight_layout()
plt.savefig('../pdf/robustness_optimizer_loss.pdf', format='pdf')
plt.show()

# 绘制第二组数据
plt.plot(acc1['extract_acc'], color=color[0], label='Adam')
plt.plot(acc2['extract_acc'], color=color[1], label='RMSProp')
plt.plot(acc3['extract_acc'], color=color[2], label='SGD')
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.xlabel('Epoch')
plt.ylabel('Extraction accuracy')
plt.tick_params(axis='y')
plt.legend()
plt.tight_layout()
plt.savefig('../pdf/robustness_optimizer_acc.pdf', format='pdf')
plt.show()
