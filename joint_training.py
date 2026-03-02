"""
文件名: joint_training.py
作者: 徐辰屹
日期: 2024年3月6日

说明:
联合训练文件，用于训练编码器与解码器
运行将会生成对应模型文件
一般任务模型的训练不会开启
该文件一次初始化一整个模型的所有参数
一般默认不开启模型训练，直接将生成的带有秘密信息的参数交给解码器
"""
import argparse
import random
import torch
from torch.utils.data import TensorDataset, DataLoader

from get_data import get_cifar10_data
from stego_model import *
from model import *
from test import test_model
from train import train_model
from utils import count_parameters, get_model_params, to_hist_tensor, modify_distribution, \
     get_secretbits_for_train, interpolate

parser = argparse.ArgumentParser(description='。。。')
parser.add_argument('--max_nums', default=500000, type=int,
                    help='最大参数提取数(如果某个层的参数个数超过这个数的参数不生成也不提取)')
parser.add_argument('--min_nums', default=1000, type=int,
                    help='最大参数提取数(如果某个层的参数个数不足这个数的参数不生成也不提取)')
parser.add_argument('--var', default=1, type=float, help='生成参数的方差(含秘模型所需要服从的方差)')
parser.add_argument('--simulation_train', default=True, type=bool, help='开启后直接在任务模型中添加噪声，模拟模型训练')
parser.add_argument('--simulation_std', default=1, type=float, help='模拟训练的方差(不宜过大)')
parser.add_argument('--simulation_mean', default=0, type=float, help='模拟训练的均值(不宜过大)')
parser.add_argument('--load_model', default=False, type=bool, help='是否加载模型')
parser.add_argument('--size', default=96, type=bool, help='每多少个秘密信息生成1024个参数')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()

if args.load_model:
    secret_bits_encoder = torch.load(f"data/encoder.pth")
    secret_bits_decoder = torch.load(f"data/decoder.pth")
else:
    secret_bits_encoder = SecretBitsEncoder(args.size).to(device)
    secret_bits_decoder = SecretBitsDecoder(args.size).to(device)

print(secret_bits_decoder)

parameters = list(secret_bits_encoder.parameters()) + list(secret_bits_decoder.parameters())

criterion_decoder = torch.nn.MSELoss()
optimizer_decoder = torch.optim.Adam(parameters, lr=5e-5)

if not args.simulation_train:
    train_loader, test_loader = get_cifar10_data()

task_model = ResNet18()
params = get_model_params(task_model)

# 删除 numel 大于 500000 的张量
# 如果你的显存够大, 也可以不删
params = [tensor for tensor in params if tensor.numel() <= 500000]

# 直接用模型的所有参数进行训练
# 参数越多相当于batchsize越大
params = torch.concatenate(params)

# 计算生成参数的直方图概率分布, 用于计算 kl 散度
bins = int(math.sqrt(len(params)))
hist_tensor, bin_center1 = to_hist_tensor(params, bins)

loss_list = []
kl_list = []
epoch = 5000
for i in range(0, epoch):

    # 任务模型的参数个数
    secret_bits = get_secretbits_for_train(len(params), size=args.size)
    secret_bits = secret_bits.to(device)

    # 生成含有秘密信息的参数
    orignal_params = secret_bits_encoder(secret_bits.to(device))
    # 对生成的参数使用最大池化进行裁剪
    orignal_params = F.adaptive_max_pool1d(orignal_params.view(1, -1), len(params)).view(-1)
    # 修改生成参数的方差
    orignal_params = modify_distribution(orignal_params, args.var)

    hist_params, bin_center2 = to_hist_tensor(orignal_params, bins)
    kl_divergence = F.kl_div(hist_tensor.log(), hist_params, reduction='sum')

    if args.simulation_train:
        # 如果开启模拟训练, 则训练噪声为随机生成的数据
        random_integer = random.randrange(2)
        if random_integer == 0:
            inaccuracies = torch.normal(args.simulation_mean, args.simulation_std, orignal_params.size()).to(device)
        else:
            inaccuracies = torch.normal(-args.simulation_mean, args.simulation_std, orignal_params.size()).to(device)
    else:
        model_params = orignal_params.detach().clone()
        point = 0
        for name, m in task_model.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                # 如果参数过多则不生成参数
                # 初始化模型参数
                params_num_perlayer = count_parameters(m)
                if params_num_perlayer > args.max_nums or params_num_perlayer < args.min_nums:
                    continue
                if m.bias is None:
                    m.weight = nn.Parameter(model_params[point:point + params_num_perlayer].reshape(m.weight.shape))
                else:
                    m.bias = nn.Parameter(model_params[point:point + m.bias.numel()])
                    m.weight = nn.Parameter(
                        model_params[point + m.bias.numel():point + params_num_perlayer].reshape(m.weight.shape))

                point += params_num_perlayer
        optimizer = torch.optim.Adam(task_model.parameters(), lr=1e-4)
        train_model(task_model, train_loader, criterion, optimizer, num_epochs=10)
        test_model(task_model, test_loader, criterion)

        # 获取模型参数
        last_params = torch.concatenate(get_model_params(task_model)).to(device)
        # 训练噪声
        inaccuracies = (last_params - orignal_params).detach()


    # 对生成的参数添加噪声()
    orignal_params = orignal_params + inaccuracies

    # 修改获取到的影写模型的方差，使其符合神经网络的输入
    orignal_params = modify_distribution(orignal_params, var=1).view(1, 1, -1)
    # 对参数进行线性插值至 1024 的倍数
    orignal_params = interpolate(orignal_params).view(-1, 1024)

    outputs = secret_bits_decoder(orignal_params)

    # 计算损失
    loss = criterion_decoder(secret_bits, outputs)
    optimizer_decoder.zero_grad()
    loss.backward()
    optimizer_decoder.step()

    predictions = (outputs > 0.5).float()
    correct = (predictions == secret_bits).sum().item()
    accuracy = correct / secret_bits.numel()
    loss_list.append(loss.item())
    kl_list.append(kl_divergence)
    print(f"epoch:{i}, loss:{loss.item()}, acc:{accuracy}, kl:{kl_divergence}")

# torch.save(loss_list, f"data/joint_train_loss.pth")
# torch.save(kl_list, f"data/joint_train_kl.pth")
torch.save(secret_bits_encoder, f"models/encoder{args.size}.pth")
torch.save(secret_bits_decoder, f"models/decoder{args.size}.pth")
