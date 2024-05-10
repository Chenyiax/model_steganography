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

from get_data import get_cnn_data
from task_model import *
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
parser.add_argument('--simulation_train', default=True, type=bool, help='开启后直接在任务模型中添加噪声，模拟模型训练')
parser.add_argument('--simulation_std', default=0.06, type=float, help='模拟训练的方差(不宜过大)')
parser.add_argument('--simulation_mean', default=0.0027, type=float, help='模拟训练的均值(不宜过大)')
parser.add_argument('--load_model', default=False, type=bool, help='是否加载模型')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()

if args.load_model:
    secret_bits_encoder = torch.load(f"data/encoder.pth")
    secret_bits_decoder = torch.load(f"data/decoder.pth")
else:
    secret_bits_encoder = SecretBitsEncoder().to(device)
    secret_bits_decoder = SecretBitsDecoder().to(device)

print(secret_bits_decoder)

parameters = list(secret_bits_encoder.parameters()) + list(secret_bits_decoder.parameters())

criterion_decoder = torch.nn.MSELoss()
optimizer_decoder = torch.optim.Adam(parameters, lr=5e-5)

if not args.simulation_train:
    train_loader, test_loader = get_cnn_data()

task_model = ResNet18()
params = get_model_params(task_model, max_nums=args.max_nums, min_nums=args.min_nums)
# 直接用模型的所有参数进行训练
# 参数越多相当于batchsize越大
params = torch.concatenate(params)

var = 1e-3
bins = int(math.sqrt(len(params)))
hist_tensor, bin_center1 = to_hist_tensor(params, bins)

loss_list = []
kl_list = []
epoch = 5000
for i in range(0, epoch):

    # 任务模型的参数个数
    secret_bits = get_secretbits_for_train(len(params))
    secret_bits = secret_bits.to(device)

    orignal_params = secret_bits_encoder(secret_bits.to(device), len(params), var)

    hist_params, bin_center2 = to_hist_tensor(orignal_params, bins)
    kl_divergence = F.kl_div(hist_tensor.log(), hist_params, reduction='sum')

    if args.simulation_train:
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
        last_params = torch.concatenate(get_model_params(task_model, max_nums=args.max_nums, min_nums=args.min_nums)).to(device)
        # 训练噪声
        inaccuracies = (last_params - orignal_params).detach()


    # 对生成的参数添加噪声()
    orignal_params = orignal_params + inaccuracies

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
torch.save(secret_bits_encoder, f"data/encoder.pth")
torch.save(secret_bits_decoder, f"data/decoder.pth")
