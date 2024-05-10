import random

import bchlib
import numpy as np
import torch

from utils import bytearray_to_int_list


# def rs_encode(dataset: np.ndarray, rs_bytes: int) -> np.ndarray:
#     output_list = []
#     rs = reedsolo.RSCodec(rs_bytes)
#     for i in dataset:
#         data_clip = i.astype(bool)
#         byte_stream = bytes(np.packbits(data_clip))
#         # 编码数据
#         encoded_data = rs.encode(byte_stream)
#         output_list.append(bytearray_to_bool_list(encoded_data))
#     return np.array(output_list, dtype=int)
#
#
# def rs_decode(encoded_data: np.ndarray, rs_bytes: int) -> torch.Tensor:
#     output_list = []
#     rs = reedsolo.RSCodec(rs_bytes)
#     for i in encoded_data:
#         data_clip = i.astype(bool)
#         byte_stream = bytes(np.packbits(data_clip))
#         decoded_data, _, _ = rs.decode(byte_stream)
#         output_list.append(bytearray_to_bool_list(decoded_data))
#     return torch.tensor(output_list, dtype=torch.int)


def random_flip(int_list, n):
    """
    随机翻转列表中的 n 个元素

    参数:
    int_list (list): 包含 0 和 1 的整数列表
    n (int): 需要翻转的元素个数
    """
    # 确保 n 不超过列表长度
    n = min(n, len(int_list[0]))

    # 随机选择 n 个索引
    flip_indices = random.sample(range(len(int_list[0])), n)

    # 对选择的索引进行翻转
    for index in flip_indices:
        int_list[0][index] = 1 if int_list[0][index] == 0 else 0

    return int_list


def bch_encode(data: np.ndarray):
    bch = bchlib.BCH(10, m=7)
    outputs = []
    for i in data:
        data_clip = i.astype(bool)
        byte_stream = bytearray(np.packbits(data_clip))
        ecc = bytearray(bch.encode(byte_stream))
        encoded_data = bytearray_to_int_list(byte_stream + ecc)
        outputs.append(encoded_data)
    return np.array(outputs)

def bch_decode(data: np.ndarray):
    bch = bchlib.BCH(10, m=7)
    outputs = []
    for i in data:
        data_clip = i.astype(bool)
        byte_stream = bytearray(np.packbits(data_clip))
        data, ecc = byte_stream[:-bch.ecc_bytes], byte_stream[-bch.ecc_bytes:]
        nerr = bch.decode(data, ecc)
        bch.correct(data, ecc)
        decoded_data = bytearray_to_int_list(data)
        outputs.append(decoded_data)
    return np.array(outputs)


bch = bchlib.BCH(10, m=7)
print(bch.ecc_bits)
for i in range(0, 1000):
    random_binary = np.random.randint(0, 2, size=(1, 56))

    encoded_data = bch_encode(random_binary.copy())

    filped_data = random_flip(encoded_data.copy(), 10)

    output = bch_decode(np.array(filped_data))

    different_elements1 = np.sum(random_binary[0] != output[0])
    different_elements2 = np.sum(encoded_data[0] != filped_data[0])

    print(different_elements1, different_elements2)