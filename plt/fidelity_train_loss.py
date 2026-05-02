import torch
import matplotlib.pyplot as plt

from plt.mpl_config import set_style

color = set_style()

model_list = ["AlexNet", "DenseNet", "ResNet18", "Vgg16"]
dataset_list = ["cifar10", "mnist", "fashionmnist"]

for dataset in dataset_list:
    for model in model_list:
        # 加载损失数据
        loss1 = torch.load(f"../data/train_loss_{model}_without_secret_{dataset}.pth")
        loss2 = torch.load(f"../data/train_loss_{model}_with_secret_{dataset}.pth")

        # 绘制第一个图：AlexNet的损失
        plt.figure()
        plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
        plt.plot(loss1, color=color[0], label='Clean')
        plt.plot(loss2, color=color[1], label='Stego')
        plt.legend()

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.tight_layout()
        plt.savefig(f'../pdf/fidelity_loss_{model}_{dataset}.pdf', dpi=None, format='pdf')
        plt.show()