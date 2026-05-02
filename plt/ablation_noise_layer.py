import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

from plt.mpl_config import set_style

color = set_style()

prune_rates = np.linspace(0, 100, 100)

data = torch.load(f"../data/extract_acc_with_noise.pth")
extract_acc_bch_list1 = data['extract_acc_bch']
data = torch.load(f"../data/extract_acc_without_noise.pth")
extract_acc_bch_list2 = data['extract_acc_bch']
plt.plot(prune_rates, extract_acc_bch_list2, color=color[1], label=f"w/o noise layer")
plt.plot(prune_rates, extract_acc_bch_list1, color=color[0], label=f"w/ noise layer")
plt.xlabel("Epoch")
plt.ylabel("Extract accuracy")

plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))

plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
plt.tight_layout()
plt.savefig('../pdf/ablation_noise_layer.pdf', format='pdf')
plt.show()