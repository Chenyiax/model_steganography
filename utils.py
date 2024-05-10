"""
文件名: utils.py
作者: 徐辰屹
日期: 2024年3月18日

说明:
工具文件
提供了一些项目中需要使用的工具函数
"""
import math
import random

import bchlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.nn.init import _calculate_correct_fan, calculate_gain


# 统计模型所有参数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 生成秘密信息的函数
def get_secretbits(nums: int) -> (torch.Tensor, torch.Tensor):
    batch = nums // 1024 + 1
    random_binary = np.random.randint(0, 2, size=(batch, 56))
    binary = bch_encode(random_binary)
    return torch.tensor(binary, dtype=torch.float32), torch.tensor(random_binary, dtype=torch.float32).view(-1)


def get_secretbits_for_train(nums: int) -> torch.Tensor:
    batch = nums // 1024 + 1
    random_binary = np.random.randint(0, 2, size=(batch, 128))
    return torch.tensor(random_binary, dtype=torch.float32)


# 将tensor进行下采样
def downsample_tensor(tensor, target_length=1024):
    tensor = tensor.reshape(1, -1)
    downsampled_tensor = torch.nn.functional.interpolate(tensor.unsqueeze(0),
                                                         size=target_length,
                                                         mode='linear',
                                                         align_corners=False).reshape(-1)
    return downsampled_tensor


def get_model_params(model: torch.nn.Module, max_nums=5000000, min_nums=1000) -> list:
    last_params_list = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            params = count_parameters(m)
            if params > max_nums or params < min_nums:
                continue
            # 获取更新后的模型参数
            if m.bias is None:
                last_params_list.append(m.weight.detach().reshape(-1).to("cpu"))
            else:
                last_params_list.append(torch.concatenate([m.bias.detach(), m.weight.detach().reshape(-1)]).to("cpu"))
    return last_params_list


def to_hist_tensor(tensor: torch.Tensor, bins: int) -> (torch.Tensor, np.ndarray):
    """
    将张量转换为直方图的函数

    参数：
    tensor(torch.Tensor) :输入张量
    bins(int) :直方图个数

    返回值：
    tensor: 张量的直方图
    ndarray: 对应直方图所代表的横坐标, 用于绘图
    """
    hist, bin_edges = np.histogram(tensor.detach().to("cpu").numpy(), bins=bins, range=(-1, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    prob_dist = hist / hist.sum()
    prob_dist[prob_dist == 0] = 1e-6
    prob_dist = torch.tensor(prob_dist, dtype=torch.float32)
    return prob_dist, bin_centers


def calculate_kl(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """
    计算两个张量的kl散度的函数
    使用直方图估计法

    参数：
    tensor1(torch.Tensor) :张量1
    tensor2(torch.Tensor) :张量2

    返回值：
    float: 两个张量的 kl 散度
    """
    # 直方图箱子的个数为参数个数的根号
    bins = int(math.sqrt(len(tensor1)))
    hist_tensor1, bin_center1 = to_hist_tensor(tensor1, bins=bins)
    hist_tensor2, bin_center2 = to_hist_tensor(tensor2, bins=bins)

    # 计算 KL 散度
    kl_divergence = F.kl_div(hist_tensor1.log(), hist_tensor2, reduction='sum')

    # 绘制概率分布曲线
    plt.figure(figsize=(10, 5))
    plt.plot(bin_center1, hist_tensor1, label='without secret')
    plt.plot(bin_center2, hist_tensor2, label='with secret')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.title('Histogram of Tensors')
    plt.legend()
    plt.show()

    return kl_divergence.item()


def modify_distribution(tensor: torch.Tensor, var: float, mean=0) -> torch.Tensor:
    """
    修改张量均值和方差的函数

    参数：
    tensor(torch.Tensor) :目标张量
    var(float) :目标方差
    mean(int) :目标均值

    返回值：
    torch.Tensor: 修改后的张量
    """
    # 计算当前张量的均值和方差
    current_mean = torch.mean(tensor)
    current_var = torch.var(tensor)

    # 计算调整因子
    adjustment_factor = torch.sqrt(var / current_var)

    # 调整张量的元素以满足指定的均值和方差
    return (tensor - current_mean) * adjustment_factor + mean


def interpolate(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.view(1, 1, -1)
    nums = tensor.size(2)
    batch = nums // 1024 + 1
    target = batch * 1024
    scale_factor = target / nums + 1e-9
    tensor = F.interpolate(tensor, scale_factor=scale_factor, mode='linear',
                           align_corners=False)
    return tensor.view(-1)


def compute_accuracy(predictions, targets):
    # 将预测的概率值转换为类别标签（0或1）
    predicted_labels = torch.argmax(predictions, dim=1)
    # 计算预测正确的样本数
    correct_predictions = torch.sum(predicted_labels == targets).item()
    # 计算准确率
    accuracy = correct_predictions / len(targets)
    return accuracy


# 字节转比特流的函数
def bytearray_to_int_list(byte_array):
    int_list = []
    for byte in byte_array:
        # 将每个字节拆分为8位，并将每位转换为整数
        for i in range(7, -1, -1):
            int_list.append((byte >> i) & 1)
    return int_list


def bch_encode(data: np.ndarray) -> np.ndarray:
    outputs = []
    bch = bchlib.BCH(10, m=7)
    for i in data:
        data_clip = i.astype(bool)
        byte_stream = bytearray(np.packbits(data_clip))
        ecc = bytearray(bch.encode(byte_stream))
        encoded_data = bytearray_to_int_list(byte_stream + ecc)
        outputs.append(encoded_data)
    del bch
    return np.array(outputs)


def bch_decode(data: np.ndarray) -> torch.Tensor:
    outputs = []
    bch = bchlib.BCH(10, m=7)
    for i in data:
        data_clip = i.astype(bool)
        byte_stream = bytearray(np.packbits(data_clip))
        data, ecc = byte_stream[:-bch.ecc_bytes], byte_stream[-bch.ecc_bytes:]
        nerr = bch.decode(data, ecc)
        bch.correct(data, ecc)
        decoded_data = bytearray_to_int_list(data)
        outputs.append(decoded_data)
    del bch
    return torch.tensor(outputs)


def kaiming_uniform_(
    tensor: torch.Tensor, a: float = math.sqrt(3), mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'
):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return std