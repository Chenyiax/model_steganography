"""
文件名: init_function.py
作者: 徐辰屹
日期: 2024年5月28日

说明:
初始化函数文件
提供了各种载体模型的初始化方法
"""
import math

from torch import nn
from torch.nn import init

from .util import kaiming_init_


def init_alexnet(m: nn.Module):
    '''
    alexnet的初始化方法，详情请见: torchvision.models.alexnet

    Args:
        m: pytorch模型的某一层

    Returns:
        weight_var: 这层权重初始化时所需要的方差
        bias_var: 这层偏置初始化时所需要的方差
    '''
    if isinstance(m, nn.Linear):
        weight_var, bias_var = kaiming_init_(m.weight, a=math.sqrt(5))
    elif isinstance(m, nn.Conv2d):
        weight_var, bias_var = kaiming_init_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                bias_var = bound**2 / 3
    else:
        weight_var = 1
        bias_var = 0
    return weight_var, bias_var


def init_densenet(m: nn.Module):
    '''
    densenet的初始化方法，详情请见: torchvision.models.densenet

    Args:
        m: pytorch模型的某一层

    Returns:
        weight_var: 这层权重初始化时所需要的方差
        bias_var: 这层偏置初始化时所需要的方差
    '''
    if isinstance(m, nn.Linear):
        weight_var, _ = kaiming_init_(m.weight, a=math.sqrt(5))
        bias_var = 0
    elif isinstance(m, nn.Conv2d):
        weight_var, bias_var = kaiming_init_(m.weight)
    else:
        weight_var = 1
        bias_var = 0
    return weight_var, bias_var


def init_resnet(m: nn.Module):
    '''
    resnet18的初始化方法，详情请见: torchvision.models.resnet

    Args:
        m: pytorch模型的某一层

    Returns:
        weight_var: 这层权重初始化时所需要的方差
        bias_var: 这层偏置初始化时所需要的方差
    '''
    if isinstance(m, nn.Linear):
        weight_var, bias_var = kaiming_init_(m.weight, a=math.sqrt(5))
    elif isinstance(m, nn.Conv2d):
        weight_var, bias_var = kaiming_init_(m.weight, mode="fan_out", nonlinearity="relu")
    else:
        weight_var = 1
        bias_var = 0
    return weight_var, bias_var



def init_vgg(m: nn.Module):
    '''
    vgg16的初始化方法，详情请见: torchvision.models.vgg

    Args:
        m: pytorch模型的某一层

    Returns:
        weight_var: 这层权重初始化时所需要的方差
        bias_var: 这层偏置初始化时所需要的方差
    '''
    if isinstance(m, nn.Linear):
        weight_var = 8.2262e-05
        bias_var = 0
    elif isinstance(m, nn.Conv2d):
        weight_var, _ = kaiming_init_(m.weight, mode="fan_out", nonlinearity="relu")
        bias_var = 0
    else:
        weight_var = 1
        bias_var = 0
    return weight_var, bias_var


def init_vit(m: nn.Module):
    '''
    visiontransformer的初始化方法，详情请见: torchvision.models.vision_transformer

    Args:
        m: pytorch模型的某一层

    Returns:
        weight_var: 这层权重初始化时所需要的方差
        bias_var: 这层偏置初始化时所需要的方差
    '''
    if isinstance(m, nn.Linear):
        weight_var, bias_var = kaiming_init_(m.weight, a=math.sqrt(5))
    elif isinstance(m, nn.Conv2d):
        weight_var, bias_var = kaiming_init_(m.weight)
    else:
        weight_var = 1
        bias_var = 0

    if m.__class__.__name__ == 'conv_proj' and isinstance(m, nn.Conv2d):
        fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
        weight_var = math.sqrt(math.sqrt(1 / fan_in))
        bias_var = 0

    elif m.__class__.__name__ == 'conv_last' and isinstance(m, nn.Conv2d):
        weight_var = math.sqrt(math.sqrt(2.0 / m.out_channels))
        bias_var = 0
    if m.__class__.__name__ == "pre_logits" and isinstance(m, nn.Linear):
        fan_in = m.in_features
        weight_var = math.sqrt(math.sqrt(1 / fan_in))
        bias_var = 0

    if m.__class__.__name__ == "heads" and isinstance(m, nn.Linear):
        weight_var = 0
        bias_var = 0

    return weight_var, bias_var


def init_nlp(m: nn.Module):
    '''
    nlp 模型的初始化方法, 但是 nlp 没有 torchvision 这么方便的东西,所以就用了 pytorch 自带的初始化方法

    Args:
        m: pytorch模型的某一层

    Returns:
        weight_var: 这层权重初始化时所需要的方差
        bias_var: 这层偏置初始化时所需要的方差
    '''
    if isinstance(m, nn.Linear):
        weight_var, bias_var = kaiming_init_(m.weight, a=math.sqrt(5))
    elif isinstance(m, nn.Conv2d):
        weight_var, bias_var = kaiming_init_(m.weight)
    else:
        weight_var = 1
        bias_var = 0
    return weight_var, bias_var