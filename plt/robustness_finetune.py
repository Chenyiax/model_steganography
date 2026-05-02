import numpy as np
import torch
from matplotlib import pyplot as plt

from plt.mpl_config import set_style

color = set_style()


data = torch.load("../data/fine-tuning_1e-5.pth")
acc_list = data['acc_list']
acc_list_bch = data['acc_list_bch']

prune_rates = np.linspace(0, 100, 100)
plt.plot(prune_rates, acc_list, color=color[0], label="w/o BCH")
plt.plot(prune_rates, acc_list_bch, color=color[1], label="w/ BCH")
plt.xlabel("Fine-tuning epoch")
plt.ylabel("Extraction accuracy")
plt.legend()
plt.ylim(0.95, 1.003)
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.tight_layout()
plt.savefig('../pdf/robustness_finetune_1e-5.pdf', format='pdf')
plt.show()

data = torch.load("../data/fine-tuning_5e-5.pth")
acc_list = data['acc_list']
acc_list_bch = data['acc_list_bch']

plt.plot(prune_rates, acc_list, color=color[0], label="w/o BCH")
plt.plot(prune_rates, acc_list_bch, color=color[1], label="w/ BCH")
plt.xlabel("Fine-tuning epoch")
plt.ylabel("Extraction accuracy")
plt.legend()
plt.ylim(0.95, 1.003)
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.tight_layout()
plt.savefig('../pdf/robustness_finetune.py_5e-5.pdf', format='pdf')
plt.show()