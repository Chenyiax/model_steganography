'''
File name: imperceptibility_kl.py
Description:
Plot the KL divergence fitting degree
'''
import torch
import torch.nn.functional as F
import argparse

import sys
import os

# Ensure the parent directory is in the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.init_function import *
from utils.util import to_hist_tensor
import matplotlib.pyplot as plt
from plt.mpl_config import set_style

def main():
    parser = argparse.ArgumentParser(description="Analyze model parameter KL divergence")
    parser.add_argument('--model', type=str, default="AlexNet", choices=["AlexNet", "Vgg16", "Resnet18", "DenseNet"], help="Model name")
    args = parser.parse_args()

    model_name = args.model
    
    init_func_map = {
        "AlexNet": init_alexnet,
        "Vgg16": init_vgg,
        "Resnet18": init_resnet,
        "DenseNet": init_densenet
    }
    model_layer_map = {
        "AlexNet": 0,
        "Vgg16": 2,
        "Resnet18": 5,
        "DenseNet": 15
    }

    init_function = init_func_map[model_name]
    
    model_path_base = f"../stego_models/{model_name}_cifar10"
    model1 = torch.load(f"{model_path_base}_with_secret.pth")
    model2 = torch.load(f"{model_path_base}_without_secret.pth")

    color = set_style()

    params_with_secret = []
    params_without_secret = []

    with torch.no_grad():  # Disable gradient calculation
        for name, m in model1.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Embedding)):
                weight_var, bias_var = init_function(m)

                # Count the number of parameters in this layer
                if hasattr(m, 'bias') and m.bias is not None and bias_var > 1e-4:
                    params_num = m.weight.numel() + m.bias.numel()
                else:
                    params_num = m.weight.numel()

                # Skip if there are too many or too few parameters
                if params_num > 500000 or params_num < 1000:
                    continue

                # Skip if the variance is too small to embed secret information
                if weight_var < 1e-4:
                    continue

                if m.bias is None:
                    params_with_secret.append(m.weight.detach().reshape(-1).to("cpu"))
                else:
                    params_with_secret.append(torch.concatenate([m.bias.detach(), m.weight.detach().reshape(-1)]).to("cpu"))

    with torch.no_grad():  # Disable gradient calculation
        for name, m in model2.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Embedding)):
                weight_var, bias_var = init_function(m)

                # Count the number of parameters in this layer
                if hasattr(m, 'bias') and m.bias is not None and bias_var > 1e-4:
                    params_num = m.weight.numel() + m.bias.numel()
                else:
                    params_num = m.weight.numel()

                # Skip if there are too many or too few parameters
                if params_num > 500000 or params_num < 1000:
                    continue

                # Skip if the variance is too small to embed secret information
                if weight_var < 1e-4:
                    continue
                if m.bias is None:
                    params_without_secret.append(m.weight.detach().reshape(-1).to("cpu"))
                else:
                    params_without_secret.append(torch.concatenate([m.bias.detach(), m.weight.detach().reshape(-1)]).to("cpu"))

    k = 0
    kl_list = []
    for i, j in zip(params_without_secret, params_with_secret):
        bins = int(math.sqrt(len(i)))
        hist_tensor1, bin_centers1 = to_hist_tensor(i, bins)
        hist_tensor2, bin_centers2 = to_hist_tensor(j, bins)

        kl_divergence = F.kl_div(hist_tensor1.log(), hist_tensor2, reduction='sum')
        kl_list.append(kl_divergence)



        plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.1)
        plt.plot(bin_centers1, hist_tensor1, color=color[1], label='Clean')
        plt.plot(bin_centers2, hist_tensor2, color=color[0], label='Stego')
        plt.xlabel('Parameter values')
        plt.ylabel('Frequency')
        plt.title(f"KL = {kl_divergence:.4f}")
        plt.legend()
        plt.tight_layout()
        if k == model_layer_map[model_name]:
            plt.savefig(f'../pdf/imperceptibility_kl_{model_name}_{k}.pdf', format='pdf')
        k += 1
        plt.close()

    kl_list = torch.tensor(kl_list)
    print(torch.mean(kl_list))

if __name__ == '__main__':
    main()