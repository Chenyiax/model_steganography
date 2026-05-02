import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.utils import prune

from utils.init_function import init_resnet
from model_steganography import ModelSteganography
from utils.get_data import get_cifar10_data
from utils.test import test_model

init_func = init_resnet
ms = ModelSteganography(init_func, target_var=1e-4, max_nums=500000)
secrets = torch.load("data/secret_ResNet18")
secret_bits = secrets["secret_bits"]
secret_bits_bch = secrets["secret_bits_bch"]
acc_list = []
acc_list_bch = []

model_acc_list = []

train_loader, test_loader = get_cifar10_data()
criterion = torch.nn.CrossEntropyLoss()
for i in range(0, 100):
    model = torch.load("models/ResNet18_with_secret.pth")

    prune_ratio = 0.01 * i

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Embedding)):
            weight = module.weight.data.cpu().numpy()
            total_weights = weight.size
            num_prune = int(total_weights * prune_ratio)

            # 基于权重绝对值的大小排序
            flat_weight = weight.flatten()
            sorted_indices = np.argsort(np.abs(flat_weight))
            prune_indices = sorted_indices[:num_prune]

            # 随机选择要剪枝的索引
            # prune_indices = np.random.choice(total_weights, num_prune, replace=False)

            flat_weight = weight.flatten()
            flat_weight[prune_indices] = 0
            module.weight.data = torch.tensor(flat_weight.reshape(weight.shape), dtype=module.weight.data.dtype).to(
                module.weight.data.device)

    model_acc = test_model(model, test_loader, criterion)
    print(model_acc)
    model_acc_list.append(model_acc)

    outputs_secrets, outputs_secrets_bch = ms.decode(model)

    correct = (outputs_secrets == secret_bits).sum().item()
    accuracy = correct / outputs_secrets.numel()
    acc_list.append(accuracy)

    correct = (outputs_secrets_bch == secret_bits_bch).sum().item()
    accuracy_bch = correct / outputs_secrets_bch.numel()
    acc_list_bch.append(accuracy_bch)
    print(accuracy, accuracy_bch)

data = {
    'acc_list': acc_list,
    'acc_list_bch': acc_list_bch,
    'model_acc_list': model_acc_list
}

# 保存字典到文件
torch.save(data, 'data/less_pruning.pth')