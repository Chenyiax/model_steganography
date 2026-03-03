'''
文件名: kl_diver.py
作者: 徐辰屹
日期: 2024年4月29日

说明:
绘制 kl 散度拟合程度
'''

import math
import torch
import torch.nn.functional as F
from torch import nn

from init_function import *
from utils.util import to_hist_tensor
import matplotlib.pyplot as plt

models = ["AlexNet", "Vgg16", "Resnet18", "DenseNet"]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({'font.size': 18})
cmap = plt.get_cmap('bwr') # bwr 色组
color1 = cmap(0)  # 取出蓝色
color2 = cmap(255)  # 取出中间色（白色和红色的中间）

init_function = init_alexnet

model1 = torch.load(f"../models/AlexNet_with_secret.pth")
model2 = torch.load(f"../models/AlexNet_without_secret.pth")

params_with_secret = []
params_without_secret = []

with torch.no_grad():  # 禁用梯度计算
    for name, m in model1.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Embedding)):
            weight_var, bias_var = init_function(m)

            # 统计这层模型参数个数
            if hasattr(m, 'bias') and m.bias is not None and bias_var > 1e-4:
                params_num = m.weight.numel() + m.bias.numel()
            else:
                params_num = m.weight.numel()

            # 如果参数过多则不生成参数
            if params_num > 500000 or params_num < 1000:
                continue

            # 如果方差过小则不嵌入秘密信息
            if weight_var < 1e-4:
                continue

            if m.bias is None:
                params_with_secret.append(m.weight.detach().reshape(-1).to("cpu"))
            else:
                params_with_secret.append(torch.concatenate([m.bias.detach(), m.weight.detach().reshape(-1)]).to("cpu"))


with torch.no_grad():  # 禁用梯度计算
    for name, m in model2.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Embedding)):
            weight_var, bias_var = init_function(m)

            # 统计这层模型参数个数
            if hasattr(m, 'bias') and m.bias is not None and bias_var > 1e-4:
                params_num = m.weight.numel() + m.bias.numel()
            else:
                params_num = m.weight.numel()

            # 如果参数过多则不生成参数
            if params_num > 500000 or params_num < 1000:
                continue

            # 如果方差过小则不嵌入秘密信息
            if weight_var < 1e-4:
                continue
            if m.bias is None:
                params_without_secret.append(m.weight.detach().reshape(-1).to("cpu"))
            else:
                params_without_secret.append(torch.concatenate([m.bias.detach(), m.weight.detach().reshape(-1)]).to("cpu"))


num_params = len(params_with_secret)

kl_list = []
for i, j in zip(params_without_secret, params_with_secret):
    bins = int(math.sqrt(len(i)))
    hist_tensor1, bin_centers1 = to_hist_tensor(i, bins)
    hist_tensor2, bin_centers2 = to_hist_tensor(j, bins)

    kl_divergence = F.kl_div(hist_tensor1.log(), hist_tensor2, reduction='sum')
    kl_list.append(kl_divergence)

    plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
    plt.plot(bin_centers1, hist_tensor1, color=color1, label='Clean')
    plt.plot(bin_centers2, hist_tensor2, color=color2, label='Stego')
    plt.xlabel('Parameter values')
    plt.ylabel('Frequency')
    plt.title(f"KL = {kl_divergence:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.show()


kl_list = torch.tensor(kl_list)
print(torch.mean(kl_list))