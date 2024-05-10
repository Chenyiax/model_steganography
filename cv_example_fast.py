"""
文件名: cv_example.py
作者: 徐辰屹
日期: 2024年4月29日

说明: 一个模型嵌入秘密信息然后提取的cv示例,快速版
"""
import random

import numpy as np
import torch
from torch import nn

from model_steganorgraphy import ModelSteganography
from task_model import *
from utils import get_model_params, count_parameters

# 每 1024 个参数嵌入 128 位秘密信息

MAX_NUMS = 5000000
MIN_NUMS = 1000

task_model = DenseNet()
print(task_model)

ms = ModelSteganography()
# 生成并嵌入秘密信息
secret_bits, secret_bits_bch = ms.encode(task_model)

params = get_model_params(task_model)
i = 0
for name, m in task_model.named_modules():
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        params_num = count_parameters(m)
        if params_num > MAX_NUMS or params_num < MIN_NUMS:
            continue

        random_integer = random.randrange(2)
        if random_integer == 0:
            noise = torch.normal(0.0027, 0.03, params[i].size())
        else:
            noise = torch.normal(-0.0027, 0.03, params[i].size())

        params[i] = params[i] + noise
        if m.bias is None:
            m.weight = nn.Parameter(params[i].reshape(m.weight.shape))
        else:
            m.bias = nn.Parameter(params[i][:m.bias.numel()])
            m.weight = nn.Parameter(params[i][m.bias.numel():].reshape(m.weight.shape))
        i += 1

# 提取秘密信息
outputs_secrets, outputs_secrets_bch = ms.decode(task_model)

outputs_secrets = outputs_secrets.view(-1)
secret_bits = secret_bits.view(-1)

correct = (outputs_secrets == secret_bits).sum().item()
accuracy = correct / outputs_secrets.numel()
print("Extraction Accuracy of Secret Information:", accuracy)

correct = (outputs_secrets_bch == secret_bits_bch).sum().item()
accuracy = correct / outputs_secrets_bch.numel()

print("Extraction Accuracy of Secret Information after BCH:", accuracy)
print("secret numel:", outputs_secrets.numel(), "bits")
print("bch secret numel:", outputs_secrets_bch.numel(), "bits")