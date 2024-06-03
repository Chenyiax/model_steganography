"""
文件名: joint_training.py
作者: 徐辰屹
日期: 2024年5月4日

说明:
绘制模型抗噪新能曲线的文件
"""
import numpy as np
import matplotlib.pyplot as plt

from stego_model import *
from model import *
from utils import get_secretbits, bch_decode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

secret_bits_encoder = torch.load(f"../data/encoder.pth")
secret_bits_decoder = torch.load(f"../data/decoder.pth")


secret_var = 1e-3      # 生成参数方差
params_num = 1000000   # 生成参数个数
noise_var_arr = np.linspace(1e-3, 0.1, 20)  # 噪声方差
snr_list = []
err_list = []
err_bch_list = []
for noise_var in noise_var_arr:
    # 获取秘密信息
    secret_bits, secret_bits_bch = get_secretbits(params_num)
    secret_bits = secret_bits.to(device)

    orignal_params = secret_bits_encoder(secret_bits.to(device))

    noise = torch.normal(0, math.sqrt(noise_var), orignal_params.size()).to(device)

    # 对生成的参数添加噪声()
    orignal_params = orignal_params + noise

    outputs = secret_bits_decoder(orignal_params).to("cpu")
    outputs = (outputs > 0.5).float()

    outputs_bch = bch_decode(outputs.detach().numpy()).view(-1)

    wrong = (outputs != secret_bits.to("cpu")).sum().item()
    err = wrong / secret_bits.numel()

    wrong = (outputs_bch != secret_bits_bch).sum().item()
    err_bch = wrong / secret_bits_bch.numel()

    snr = 10*np.log(secret_var/noise_var)/np.log(10)

    snr_list.append(snr)
    err_list.append(err)
    err_bch_list.append(err_bch)

    print(f"snr:{snr}, err:{err}, err_bch:{err_bch} ")
data_dict = {"snr": snr_list, "acc":err_list, "acc_bch":err_bch_list}

plt.plot(snr_list, err_list, label="origin model")
plt.plot(snr_list, err_bch_list, label="use bch")
plt.xlabel("SNR(dB)")
plt.ylabel("BER")
plt.legend()
plt.grid()
plt.show()


