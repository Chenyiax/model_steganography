import numpy as np
import torch
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({'font.size': 18})
cmap = plt.get_cmap('bwr')  # bwr 色组
color1 = cmap(0)  # 取出蓝色
color2 = cmap(255)  # 取出中间色（白色和红色的中间）

data = torch.load("../data/fine-tuning_1e-5.pth")
acc_list = data['acc_list']
acc_list_bch = data['acc_list_bch']

prune_rates = np.linspace(0, 100, 100)
plt.plot(prune_rates, acc_list, color=color1, label="w/o BCH")
plt.plot(prune_rates, acc_list_bch, color=color2, label="w/ BCH")
plt.xlabel("Fine-tuning epoch")
plt.ylabel("Extraction accuracy")
plt.legend()
plt.ylim(0.95, 1.003)
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.tight_layout()
plt.savefig('../data/fine-tuning_1e-5.pdf', format='pdf')
plt.show()

data = torch.load("../data/fine-tuning_5e-5.pth")
acc_list = data['acc_list']
acc_list_bch = data['acc_list_bch']

plt.plot(prune_rates, acc_list, color=color1, label="w/o BCH")
plt.plot(prune_rates, acc_list_bch, color=color2, label="w/ BCH")
plt.xlabel("Fine-tuning epoch")
plt.ylabel("Extraction accuracy")
plt.legend()
plt.ylim(0.95, 1.003)
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.tight_layout()
plt.savefig('../data/fine-tuning_5e-5.pdf', format='pdf')
plt.show()