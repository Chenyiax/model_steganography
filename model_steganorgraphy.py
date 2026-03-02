"""
文件名: model_steganorgraphy.py
作者: 徐辰屹
日期: 2024年5月3日

说明:
模型隐写类
编码函数和解码函数
"""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

from utils import get_secretbits, bch_decode, modify_distribution, interpolate


class ModelSteganography:
    def __init__(self, init_function, size=128, batch_size=64, target_var=1e-3, min_nums=1000):
        '''
        :param init_function: 需要使用的初始化方法
        :param batch_size: 编码器生成参数时的 batch_size
        :param target_var: 最小方差提取数量
        :param min_nums: 最小参数提取数量
        '''
        self.size = size
        self.batch_size = batch_size
        self.target_var = target_var
        self.min_nums = min_nums
        self.init_function = init_function

    def encode(self, model: torch.nn.Module) -> (torch.Tensor, torch.Tensor):
        '''
        编码函数, 给目标模型生成携带有秘密信息的参数
        需要项目目录中有 encoder 才能使用

        Args:
             model (torch.nn.Module): 待生成带有秘密信息参数的模型

        Returns:
            torch.Tensor: 嵌入的秘密信息
            torch.Tensor: bch解码的秘密信息
        '''
        secret_bits_encoder = torch.load(f"models/encoder{self.size}.pth").train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        secret_bits_bch_arr = []  # 存储秘密信息用作验证
        secret_bits_arr = []
        with torch.no_grad():  # 禁用梯度计算
            for name, m in model.named_modules():
                if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Embedding)):
                    # 获取这层参数需要拟合的方差
                    weight_var, bias_var = self.init_function(m)

                    # 统计这层模型参数个数
                    if hasattr(m, 'bias') and m.bias is not None and bias_var > self.target_var:
                        params_num = m.weight.numel() + m.bias.numel()
                    else:
                        params_num = m.weight.numel()

                    # 如果参数过多则不生成参数
                    if params_num < self.min_nums:
                        continue

                    # 如果方差过小则不嵌入秘密信息
                    if weight_var < self.target_var:
                        continue

                    # 生成秘密信息
                    secret_bits, secret_bits_bch = get_secretbits(params_num)
                    secret_bits_bch_arr.append(secret_bits_bch)
                    secret_bits_arr.append(secret_bits)
                    # 分 batch 生成含有秘密信息的参数
                    dataset = TensorDataset(secret_bits)
                    data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
                    orignal_params_list = []
                    for batch in data_loader:
                        orignal_params = secret_bits_encoder(batch[0].to(device))
                        orignal_params_list.append(orignal_params)
                    orignal_params = torch.concatenate(orignal_params_list)
                    orignal_params = F.adaptive_max_pool1d(orignal_params.view(1, -1), params_num).view(-1)

                    # 将含有秘密信息的参数塞给模型
                    if hasattr(m, 'bias') and m.bias is not None and bias_var > self.target_var:
                        # 修改方差
                        new_bias = modify_distribution(orignal_params[0:m.bias.numel()], bias_var)
                        m.bias = nn.Parameter(new_bias)
                        new_weight = modify_distribution(orignal_params[m.bias.numel():params_num], weight_var)
                        m.weight = nn.Parameter(new_weight.reshape(m.weight.shape))
                    else:
                        orignal_params = modify_distribution(orignal_params, weight_var)
                        m.weight = nn.Parameter(orignal_params.reshape(m.weight.shape))

                # 如果是 rnn 就比较复杂
                elif isinstance(m, (nn.LSTM, nn.RNN)):
                    # 统计这层的参数
                    weight_params = {name: param for name, param in m.named_parameters() if 'weight' in name}

                    # 遍历这一层的所有参数
                    # 详情请见 pytorch 源码
                    for key, value in weight_params.items():
                        bias_name = key.replace("weight", "bias")
                        # 反射获取指定元素的个数
                        if hasattr(m, bias_name):
                            params_num = value.numel() + getattr(m, bias_name).numel()
                        else:
                            params_num = value.numel()

                        if params_num < self.min_nums:
                            continue
                        # 统计参数方差
                        var = torch.var(getattr(m, key)).item()
                        if var < self.target_var:
                            continue
                        # 生成秘密信息
                        secret_bits, secret_bits_bch = get_secretbits(params_num)
                        secret_bits_bch_arr.append(secret_bits_bch)
                        secret_bits_arr.append(secret_bits)

                        # 分 batch 生成含有秘密信息的参数
                        dataset = TensorDataset(secret_bits)
                        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
                        orignal_params_list = []
                        for batch in data_loader:
                            orignal_params = secret_bits_encoder(batch[0].to(device))
                            orignal_params_list.append(orignal_params)
                        orignal_params = torch.concatenate(orignal_params_list)
                        orignal_params = F.adaptive_max_pool1d(orignal_params.view(1, -1), params_num).view(-1)
                        orignal_params = modify_distribution(orignal_params, var)

                        # 将生成完成的参数塞给模型
                        if hasattr(m, bias_name):
                            setattr(m, bias_name, nn.Parameter(orignal_params[:getattr(m, bias_name).numel()]))
                            setattr(m, key,
                                    nn.Parameter(
                                        orignal_params[getattr(m, bias_name).numel():].reshape(getattr(m, key).shape)))
                        else:
                            setattr(m, key, nn.Parameter(orignal_params.reshape(getattr(m, key).shape)))

        # 合并为 tensor 类型
        secret_bits_bch_tensor = torch.concatenate(secret_bits_bch_arr)
        secret_bits_tensor = torch.concatenate(secret_bits_arr)
        del secret_bits_encoder
        return secret_bits_tensor.view(-1), secret_bits_bch_tensor

    def decode(self, model: torch.nn.Module) -> (torch.Tensor, torch.Tensor):
        '''
        解码函数
        需要项目目录中有 decoder 才能使用

        Args:
            model (torch.nn.Module): 待提取秘密信息的模型

        Returns:
            torch.Tensor: 嵌入的秘密信息
            torch.Tensor: bch解码的秘密信息
        '''
        secret_bits_decoder = torch.load(f"models/decoder{self.size}.pth").train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        outputs_arr_bch = []  # 存储解码出的秘密信息
        outputs_arr = []
        with torch.no_grad():  # 禁用梯度计算
            for name, m in model.named_modules():
                if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)):
                    # 获取这层参数需要拟合的方差
                    weight_var, bias_var = self.init_function(m)
                    # 统计这层需要嵌秘密信息的模型参数个数
                    if hasattr(m, 'bias') and m.bias is not None and bias_var > self.target_var:
                        params_num = m.weight.numel() + m.bias.numel()
                    else:
                        params_num = m.weight.numel()
                    # 如果参数过多则不生成参数
                    if params_num < self.min_nums:
                        continue
                    # 如果方差过小则不嵌入秘密信息
                    if weight_var < self.target_var:
                        continue
                    # 获取更新后的模型参数
                    if hasattr(m, 'bias') and m.bias is not None and bias_var > self.target_var:
                        last_params_tensor = torch.concatenate([m.bias, m.weight.reshape(-1)])
                    else:
                        last_params_tensor = m.weight.reshape(-1)

                    # 对参数进行线性插值至 1024 的倍数
                    last_params_tensor = interpolate(last_params_tensor).view(-1, 1024)

                    dataset = TensorDataset(last_params_tensor)
                    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
                    output_list = []
                    for batch in dataloader:
                        outputs = secret_bits_decoder(batch[0].to(device)).to('cpu')
                        output_list.append(outputs)
                    outputs = torch.concatenate(output_list)

                    # 大于0.5的认为是 1
                    predictions = (outputs > 0.5).float()

                    outputs_arr.append(predictions)
                    predictions = bch_decode(predictions.detach().numpy())
                    outputs_arr_bch.append(predictions)

                elif isinstance(m, (nn.LSTM, nn.RNN)):
                    weight_params = {name: param for name, param in m.named_parameters() if 'weight' in name}
                    for key, value in weight_params.items():
                        bias_name = key.replace("weight", "bias")

                        if hasattr(m, bias_name):
                            params = value.numel() + getattr(m, bias_name).numel()
                        else:
                            params = value.numel()
                        if params < self.min_nums:
                            continue

                        if hasattr(m, bias_name):
                            last_params_tensor = torch.tensor(
                                [*getattr(m, bias_name).detach().tolist(),
                                 *getattr(m, key).detach().reshape(-1).tolist()],
                                dtype=torch.float32).to(device)
                        else:
                            last_params_tensor = torch.tensor([*getattr(m, key).detach().reshape(-1).tolist()],
                                                              dtype=torch.float32).to(device)

                        last_params_tensor = modify_distribution(last_params_tensor, var=1).view(1, 1, -1)
                        # 对参数进行线性插值至 1024 的倍数
                        last_params_tensor = interpolate(last_params_tensor).view(-1, 1024)

                        # 分 batch 解码
                        dataset = TensorDataset(last_params_tensor)
                        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
                        output_list = []
                        for batch in dataloader:
                            outputs = secret_bits_decoder(batch[0].to(device)).to('cpu')
                            output_list.append(outputs)
                        outputs = torch.concatenate(output_list)

                        predictions = (outputs > 0.5).float()

                        outputs_arr.append(predictions)
                        predictions = bch_decode(predictions.detach().numpy())
                        outputs_arr_bch.append(predictions)

        outputs_tensor = torch.concatenate(outputs_arr)
        outputs_tensor_bch = torch.concatenate(outputs_arr_bch)
        return outputs_tensor.view(-1), outputs_tensor_bch.view(-1)
