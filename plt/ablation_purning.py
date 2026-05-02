import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from plt.mpl_config import set_style

color = set_style()

data1 = torch.load("../data/ablation_less_pruning_with_noise.pth")
data2 = torch.load("../data/ablation_less_pruning_without_noise.pth")
acc_list1 = data1['acc_list']
acc_list2 = data2['acc_list']

prune_rates = np.linspace(0, 100, 100)
plt.plot(prune_rates, acc_list2, color=color[1], label="w/o noise layer")
plt.plot(prune_rates, acc_list1, color=color[0], label="w/ noise layer")
plt.xlabel("Finetune epoch")
plt.ylabel("Extract accuracy")
plt.legend()

plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))

plt.ylim(0.5, 1)
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.tight_layout()
plt.savefig('../pdf/ablation_pruning.pdf', format='pdf')
plt.show()
