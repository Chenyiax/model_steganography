'''
文件名: kl_diver.py
作者: 徐辰屹
日期: 2024年4月29日

说明:
绘制对照方法的 kl 散度拟合程度
'''

import math
import torch
import numpy as np
import torch.nn.functional as F
from utils.util import to_hist_tensor
import matplotlib.pyplot as plt

params_with_secret = torch.tensor(np.load("alexnet_weight_without_secret22.npy")).view(-1)
params_without_secret = torch.tensor(np.load("alexnet_weight_without_secret.npy")).view(-1)

# 计算直方图统计的箱子个数
bins = int(math.sqrt(len(params_with_secret)))
# 转为直方图
hist_tensor1, bin_centers1 = to_hist_tensor(params_with_secret, bins)
hist_tensor2, bin_centers2 = to_hist_tensor(params_without_secret, bins)
# 计算 kl 散度
kl_divergence = F.kl_div(hist_tensor1.log(), hist_tensor2, reduction='sum')


plt.plot(bin_centers1, hist_tensor1, label='Origin')
plt.plot(bin_centers2, hist_tensor2, label='Stego')
plt.xlabel('bins')
plt.ylabel('Frequency')
plt.title(f"kl = {kl_divergence}")
plt.legend()
plt.grid(True)


plt.show()
kl_divergence = F.kl_div(hist_tensor1.log(), hist_tensor2, reduction='sum')
print(kl_divergence)
