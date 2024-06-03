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
from utils import to_hist_tensor
import matplotlib.pyplot as plt

params_with_secret = torch.load("../data/params_with_secret.pth")
params_without_secret = torch.load("../data/params_without_secret.pth")

params_without_secret = params_without_secret[0:8]
params_with_secret = params_with_secret[0:8]
num_params = len(params_with_secret)

# 计算子图的行和列数，使得子图能够容纳所有的数据
num_cols = 4
num_rows = math.ceil(num_params / num_cols)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))

for idx, (i, j) in enumerate(zip(params_with_secret, params_without_secret)):
    bins = int(math.sqrt(len(i)))
    hist_tensor1, bin_centers1 = to_hist_tensor(i, bins)
    hist_tensor2, bin_centers2 = to_hist_tensor(j, bins)

    kl_divergence = F.kl_div(hist_tensor1.log(), hist_tensor2, reduction='sum')
    row = idx // num_cols
    col = idx % num_cols
    ax = axes[row, col]

    ax.plot(bin_centers1, hist_tensor1, label='Origin')
    ax.plot(bin_centers2, hist_tensor2, label='Stego')
    ax.set_xlabel('bins')
    ax.set_ylabel('Frequency')
    ax.set_title(f"kl = {kl_divergence}")
    ax.legend()
    ax.grid(True)

# 如果剩余的子图不需要，则隐藏它们
for idx in range(num_params, num_cols * num_rows):
    row = idx // num_cols
    col = idx % num_cols
    fig.delaxes(axes[row, col])

plt.tight_layout()
plt.show()


params_without_secret = torch.concatenate(params_without_secret)
params_with_secret = torch.concatenate(params_with_secret)
bins = int(math.sqrt(len(params_without_secret)))
hist_tensor1, bin_centers1 = to_hist_tensor(params_with_secret, bins)
hist_tensor2, bin_centers2 = to_hist_tensor(params_without_secret, bins)

kl_divergence = F.kl_div(hist_tensor1.log(), hist_tensor2, reduction='sum')
print(kl_divergence)