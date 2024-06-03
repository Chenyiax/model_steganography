"""
文件名: utils.py
作者: 徐辰屹
日期: 2024年3月18日

说明:
工具文件
提供了一些项目中需要使用的工具函数
"""
import math

import bchlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.init import _calculate_correct_fan, calculate_gain


# 统计模型所有参数
def count_parameters(model, bias=True):
    if bias:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters() if p.requires_grad and p.dim() > 1)


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


def get_model_params(model: torch.nn.Module) -> list:
    '''
    获取神经网络参数的函数

    Args:
        model(nn.Module) : 目标神经网络模型

    Returns:
        list: 一个列表,长度为符合条件的层的参数， 其中每一个元素为一个tensor
        例: 一个10层的神经网络, 返回值为一个长度为 10 的 list, 每一个元素都是一个tensor, 为这一层的参数
    '''
    last_params_list = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # 获取更新后的模型参数
            if m.bias is None:
                last_params_list.append(m.weight.detach().reshape(-1).to("cpu"))
            else:
                last_params_list.append(torch.concatenate([m.bias.detach(), m.weight.detach().reshape(-1)]).to("cpu"))
    return last_params_list


def to_hist_tensor(tensor: torch.Tensor, bins: int) -> (torch.Tensor, np.ndarray):
    '''
    将张量转换为直方图的函数
    Args:
        tensor(torch.Tensor) :输入张量
        bins(int) :直方图个数

    Returns:
        tensor: 张量的直方图
        ndarray: 对应直方图所代表的横坐标, 用于绘图
    '''
    hist, bin_edges = np.histogram(tensor.detach().to("cpu").numpy(), bins=bins, range=(-1, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    prob_dist = hist / hist.sum()
    prob_dist[prob_dist == 0] = 1e-6
    prob_dist = torch.tensor(prob_dist, dtype=torch.float32)
    return prob_dist, bin_centers


def calculate_kl(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    '''
    计算两个张量的kl散度的函数
    使用直方图估计法

    Args:
        tensor1(torch.Tensor) :张量1
        tensor2(torch.Tensor) :张量2

    Returns:
         float: 两个张量的 kl 散度
    '''
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
    '''
    修改张量均值和方差的函数

    Args:
        tensor(torch.Tensor) :目标张量
        var(float) :目标方差
        mean(int) :目标均值

    Returns:
         torch.Tensor: 修改后的张量
    '''
    # 计算当前张量的均值和方差
    current_mean = torch.mean(tensor)
    current_var = torch.var(tensor)

    # 计算调整因子
    adjustment_factor = torch.sqrt(var / current_var)

    # 调整张量的元素以满足指定的均值和方差
    return (tensor - current_mean) * adjustment_factor + mean


def interpolate(tensor: torch.Tensor) -> torch.Tensor:
    '''
    将张量线性插值至 1024的倍数
    Args:
        tensor(torch.Tensor): 输入张量

    Returns:
        torch.Tensor: 输出张量
    '''
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


def bytearray_to_int_list(byte_array):
    '''
    字节转比特流的函数

    Args:
        byte_array: 字节数组

    Returns:
        list: 只含有 0 和 1 的list
    '''
    int_list = []
    for byte in byte_array:
        # 将每个字节拆分为8位，并将每位转换为整数
        for i in range(7, -1, -1):
            int_list.append((byte >> i) & 1)
    return int_list


def bch_encode(data: np.ndarray) -> np.ndarray:
    '''
    bch 编码函数
    bchlib这个库真的很阴间, 注释写太少了

    Args:
        data(np.ndarray): 只含有 0 和 1 的np数组, 长度应当为 64

    Returns:
        np.ndarray: 经过bch编码后的np数组, 只含有0和1, 长度为 128
    '''
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
    '''
    bch 解码函数
    bchlib这个库真的很阴间, 注释写太少了

    Args:
        data(np.ndarray): 只含有 0 和 1 的np数组, 长度应当为 128

    Returns:
        np.ndarray: 经过bch纠错后的np数组, 只含有0和1, 长度为 64
    '''
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


def kaiming_init_(
    tensor: torch.Tensor, a: float = math.sqrt(3), mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'
):
    '''
    凯明初始化方差计算函数
    详情请见 torch.nn.init.kaiming_uniform_

    Args:
        tensor(torch.Tensor): 待初始化的张量
        a(float): leaky_relu的斜率，如果nonlinearity是relu的话，这项参数没用
        mode(str): fan_in是优化前向传播, fan_out是优化反向传播
        nonlinearity(str): 激活函数用的是relu还是leaky_relu

    Returns:
        float: 这层权重所需要服从的方差
        float: 对应的偏置所需要服从的方差(如果有的话)
    '''
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)

    fan_in, _ = init._calculate_fan_in_and_fan_out(tensor)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    bias_var = bound**2/3
    return std**2, bias_var