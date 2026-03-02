import math

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils import to_hist_tensor, get_model_params

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({'font.size': 18})
model_init = torch.load(f"../models/ResNet18_without_secret_epoch_{0}.pth")
params_init1 = torch.concatenate(get_model_params(model_init))

model_init = torch.load(f"../models/ResNet18_without_secret_epoch_{0}_5e-5.pth")
params_init2 = torch.concatenate(get_model_params(model_init))

model_init = torch.load(f"../models/ResNet18_without_secret_epoch_{0}_1e-5.pth")
params_init3 = torch.concatenate(get_model_params(model_init))

std_list = [[], [], []]
inacc_list = [[], [], []]
for i in range(1, 100):
    model_trained = torch.load(f"../models/ResNet18_without_secret_epoch_{i}.pth")
    params_trained = torch.concatenate(get_model_params(model_trained))
    inacc = params_trained - params_init1
    inacc_list[0].append(inacc)
    std_list[0].append(torch.std(inacc))

    model_trained = torch.load(f"../models/ResNet18_without_secret_epoch_{i}_5e-5.pth")
    params_trained = torch.concatenate(get_model_params(model_trained))
    inacc = params_trained - params_init2
    inacc_list[1].append(inacc)
    std_list[1].append(torch.std(inacc))

    model_trained = torch.load(f"../models/ResNet18_without_secret_epoch_{i}_1e-5.pth")
    params_trained = torch.concatenate(get_model_params(model_trained))
    inacc = params_trained - params_init3
    inacc_list[2].append(inacc)
    std_list[2].append(torch.std(inacc))
bins = 1000
j = 0
hist_tensor_list = []
bin_center_list = []
for i in inacc_list:
    hist_tensor, bin_center = to_hist_tensor(i[-1], bins, range=(-0.2,0.2))
    hist_tensor_list.append(hist_tensor)
    bin_center_list.append(bin_center)

plt.plot(bin_center_list[0], hist_tensor_list[0], color=(0, 0, 1), label='1e-4')
plt.plot(bin_center_list[1], hist_tensor_list[1], color=(0, 1, 0), label='5e-5')
plt.plot(bin_center_list[2], hist_tensor_list[2], color=(1, 0, 0), label='1e-5')
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.xlabel('Difference of parameter values')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('../data/noise_dis.pdf', dpi=None, format='pdf')
plt.show()

plt.plot(std_list[0], color=(0, 0, 1), label='1e-4')
plt.plot(std_list[1], color=(0, 1, 0), label='5e-5')
plt.plot(std_list[2], color=(1, 0, 0), label='1e-5')
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.xlabel('Epoch')

plt.ylabel(r'$\sigma_t^2$')
plt.legend()
plt.tight_layout()
plt.savefig('../data/noise_std.pdf', dpi=None, format='pdf')
plt.show()
