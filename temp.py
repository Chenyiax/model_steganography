import torch

from init_function import *
from model_steganorgraphy import ModelSteganography
from get_data import get_cnn_data
from cover_model import *
from test import test_model
from train import train_model
from utils import get_model_params

train_loader, test_loader = get_cnn_data()
task_model = Vgg16()
param1 = get_model_params(task_model)
init_func = init_vgg
print(task_model)
# 面向对象编程
ms = ModelSteganography(init_func, target_var=1e-4, max_nums=5000000)
secret_bits, secret_bits_bch = ms.encode(task_model)
task_model.to("cpu")
param2 = get_model_params(task_model)

for i,j in zip(param1, param2):
    print(torch.var(i), torch.var(j))