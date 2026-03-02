'''
文件名: show_loss.py
作者: 徐辰屹
日期: 2024年4月29日

说明:
绘制任务模型收敛速度
'''
import torch
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
cmap = plt.get_cmap('bwr') # bwr 色组
color1 = cmap(0)  # 取出蓝色
color2 = cmap(255)  # 取出中间色（白色和红色的中间）
plt.rcParams.update({'font.size': 18})

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
        plt.plot(loss1, color=color1, label='Clean')
        plt.plot(loss2, color=color2, label='Stego')
        plt.legend()

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.tight_layout()
        plt.savefig(f'../data/{model}_loss_{dataset}.pdf', dpi=None, format='pdf')
        plt.show()