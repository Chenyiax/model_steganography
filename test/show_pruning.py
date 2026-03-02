import numpy as np
import torch
from matplotlib import pyplot as plt

data = torch.load("../data/random_pruning.pth")
acc_list = data['acc_list']
acc_list_bch = data['acc_list_bch']
model_acc_list = data['model_acc_list']

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({'font.size': 18})
cmap = plt.get_cmap('bwr')  # bwr 色组
color1 = cmap(0)  # 取出蓝色
color2 = cmap(255)  # 取出中间色（白色和红色的中间）

prune_rates = np.linspace(0, 1, 100)
plt.plot(prune_rates, acc_list, color=color1, label="w/o BCH")
plt.plot(prune_rates, acc_list_bch, color=color2, label="w/ BCH")
plt.xlabel("Pruning rate")
plt.ylabel("Extraction accuracy")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.tight_layout()
plt.savefig("../data/pruning_random.pdf", format="pdf")
plt.show()

prune_rates = np.linspace(0, 1, 100)
plt.plot(prune_rates, model_acc_list, color=color2)
plt.xlabel("Pruning rate")
plt.ylabel(r"Classification accuracy of $\mathrm {M_s}$")
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.tight_layout()
plt.savefig("../data/pruning_acc_random.pdf", format="pdf")
plt.show()

data = torch.load("../data/less_pruning.pth")
acc_list = data['acc_list']
acc_list_bch = data['acc_list_bch']
model_acc_list = data['model_acc_list']

plt.plot(prune_rates, acc_list, color=color1, label="w/o BCH")
plt.plot(prune_rates, acc_list_bch, color=color2, label="w/ BCH")
plt.xlabel("Pruning rate")
plt.ylabel("Extraction accuracy")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.tight_layout()
plt.savefig("../data/pruning_less.pdf", format="pdf")
plt.show()

prune_rates = np.linspace(0, 1, 100)
plt.plot(prune_rates, model_acc_list, color=color2)
plt.xlabel("Pruning rate")
plt.ylabel(r"Classification accuracy of $\mathrm {M_s}$")
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.tight_layout()
plt.savefig("../data/pruning_acc_less.pdf", format="pdf")
plt.show()

