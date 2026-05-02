import numpy as np
import torch
from matplotlib import pyplot as plt

from plt.mpl_config import set_style

color = set_style()

data = torch.load("../data/random_pruning.pth")
acc_list = data['acc_list']
acc_list_bch = data['acc_list_bch']
model_acc_list = data['model_acc_list']

prune_rates = np.linspace(0, 1, 100)
plt.plot(prune_rates, acc_list, color=color[0], label="w/o BCH")
plt.plot(prune_rates, acc_list_bch, color=color[1], label="w/ BCH")
plt.xlabel("Pruning rate")
plt.ylabel("Extraction accuracy")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.tight_layout()
plt.savefig("../pdf/robustness_pruning_random_extraction.pdf", format="pdf")
plt.show()

prune_rates = np.linspace(0, 1, 100)
plt.plot(prune_rates, model_acc_list, color=color[2])
plt.xlabel("Pruning rate")
plt.ylabel(r"Classification accuracy of $\mathrm {M_s}$")
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.tight_layout()
plt.savefig("../data/robustness_pruning_random_classification.pdf", format="pdf")
plt.show()

data = torch.load("../data/less_pruning.pth")
acc_list = data['acc_list']
acc_list_bch = data['acc_list_bch']
model_acc_list = data['model_acc_list']

plt.plot(prune_rates, acc_list, color=color[0], label="w/o BCH")
plt.plot(prune_rates, acc_list_bch, color=color[1], label="w/ BCH")
plt.xlabel("Pruning rate")
plt.ylabel("Extraction accuracy")
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.tight_layout()
plt.savefig("../data/robustness_pruning_less_extraction.pdf", format="pdf")
plt.show()

prune_rates = np.linspace(0, 1, 100)
plt.plot(prune_rates, model_acc_list, color=color[2])
plt.xlabel("Pruning rate")
plt.ylabel(r"Classification accuracy of $\mathrm {M_s}$")
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.tight_layout()
plt.savefig("../pdf/robustness_pruning_less_classification.pdf", format="pdf")
plt.show()

