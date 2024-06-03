"""
文件名: gan_training.py
作者: 徐辰屹
日期: 2024年3月14日

说明:
对抗训练文件，用于训练生成网络与判决器
运行将会生成对应模型文件
在训练过程中一般不开启任务模型的训练
"""
import math
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

import torch.nn.functional as F
from get_data import get_cnn_data, get_rnn_data
from cover_model import *
from model import SecretBitsEncoder, SecretBitsDecoder, Discriminator
from test import test_model
from train import train_model
from utils import count_parameters, downsample_tensor, get_model_params, get_secretbits, to_hist_tensor, \
    compute_accuracy, get_secretbits_for_train, modify_distribution

SIZE = 128
epoch = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = torch.nn.CrossEntropyLoss()

# secret_bits_encoder = SecretBitsEncoder(SIZE).to(device)
# secret_bits_decoder = SecretBitsDecoder(SIZE).to(device)
secret_bits_encoder = torch.load(f"models/encoder.pth")
secret_bits_decoder = torch.load(f"models/decoder.pth")


# discriminator = torch.load(f'discriminator_{SIZE}.pth').to(device)
discriminator = Discriminator().to(device)
parameters = list(secret_bits_encoder.parameters()) + list(secret_bits_decoder.parameters())
optimizer_encoder = torch.optim.Adam(parameters, lr=1e-4)
criterion_decoder = torch.nn.MSELoss()
# 判决器的损失
criterion_discriminator = torch.nn.CrossEntropyLoss()
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

for epoch_i in range(0, epoch):
    print("############################################")
    print(f"epoch: {epoch_i}")
    task_model = ResNet18()

    orignal_params_list = get_model_params(task_model)
    params_without_sectret = random.choice(orignal_params_list).to(device)

    var = torch.var(params_without_sectret).item()
    secret_bit = get_secretbits_for_train(len(params_without_sectret))
    secret_bit = secret_bit.to(device)
    params_with_sectret = secret_bits_encoder(secret_bit)
    # 最大池化将生成的参数池化至指定参数个数
    params_with_sectret = F.adaptive_max_pool1d(params_with_sectret.view(1, -1), len(params_without_sectret)).view(-1)
    params_with_sectret = modify_distribution(params_with_sectret, var)

    ########
    # 制作判决器需要的训练样本
    # 对模型的每一层进行参数提取,并且对每一层下采样至 1024 位
    # 将处理完成的数据交给判决器
    random_integer = random.randrange(2)
    if random_integer == 0:
        noise = torch.normal(0.0027, 0.04, params_with_sectret.size()).to(device)
    else:
        noise = torch.normal(-0.0027, 0.04, params_with_sectret.size()).to(device)

    # # 计算差值
    # inaccuracies = (last_params - orignal_params).detach()

    # 对生成的参数添加噪声
    params_without_sectret = params_without_sectret + noise
    params_with_sectret = params_with_sectret + noise

    params_to_decode = params_with_sectret.clone()

    nums = params_with_sectret.size(-1)
    batch = nums // 1024 + 1
    target = batch * 1024
    scale_factor = target / nums + 1e-9

    params_with_sectret = F.interpolate(params_with_sectret.view(1,1,-1), scale_factor=scale_factor, mode='linear',
                                        align_corners=False).view(batch, 1024).detach()
    params_without_sectret = F.interpolate(params_without_sectret.view(1,1,-1), scale_factor=scale_factor, mode='linear',
                                           align_corners=False).view(batch, 1024).detach()
    params_to_decode = F.interpolate(params_to_decode.view(1,1,-1), scale_factor=scale_factor, mode='linear',align_corners=False).view(batch, 1024)

    bins = int(math.sqrt(len(params_with_sectret.view(-1))))
    hist_tensor1, bin_center1 = to_hist_tensor(params_with_sectret.view(-1), bins=bins)
    hist_tensor2, bin_center2 = to_hist_tensor(params_without_sectret.view(-1), bins=bins)

    # 计算 KL 散度
    kl_divergence = F.kl_div(hist_tensor1.log(), hist_tensor2, reduction='sum')
    print("KL Divergence:", kl_divergence.item())

    optimizer_discriminator.zero_grad()

    dx = discriminator(params_with_sectret.detach())
    loss_real = criterion(dx, torch.ones_like(dx))

    dg = discriminator(params_without_sectret.to(device).detach())
    loss_fake = criterion(dg, torch.zeros_like(dx))

    loss_discriminator = loss_real + loss_fake
    loss_discriminator.backward()
    optimizer_discriminator.step()

    accuracy_real = compute_accuracy(dx, torch.ones_like(dx[:, 0]))
    accuracy_fake = compute_accuracy(dg, torch.zeros_like(dg[:, 0]))
    print("Accuracy of the discriminator:", (accuracy_real+accuracy_fake)/2)
    print("loss of the discriminator:", loss_discriminator.item())
    ###################
    optimizer_encoder.zero_grad()
    df = discriminator(params_to_decode)
    loss_encoder = criterion_discriminator(df, torch.ones_like(df))

    ga = secret_bits_decoder(params_to_decode)
    loss_decoder = criterion_decoder(ga, secret_bit)
    loss = loss_decoder + loss_encoder

    predictions = (ga > 0.5).float()  # 大于0.5的认为是正类
    correct = (predictions == secret_bit).sum().item()
    accuracy = correct / secret_bit.size(0) / SIZE

    loss.backward()
    optimizer_encoder.step()
    print("loss of the encoder:", loss_encoder.item())
    print("decoder acc", accuracy)

torch.save(discriminator, f"models/discriminator.pth")
torch.save(secret_bits_encoder, f"models/encoder.pth")
torch.save(secret_bits_decoder, f"models/decoder.pth")