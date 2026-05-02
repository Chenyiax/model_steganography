"""
Filename: joint_training.py
Description: Joint training script for the encoder and decoder. As opposed to adversarial training,
joint training eliminates the need for a decision maker, thereby accelerating the development of encoders and decoders
"""
import argparse
import math
import torch
import torch.nn.functional as F
import stego_model
from model import SecretBitsEncoder, SecretBitsDecoder
from utils.get_data import get_cifar10_data
from utils.train import train_model
from utils.util import count_parameters, get_model_params, to_hist_tensor, modify_distribution, get_secretbits_for_train, interpolate

def main():
    parser = argparse.ArgumentParser(description='Joint Training for Steganography')
    parser.add_argument('--max_nums', default=500000, type=int, help='Max params per layer')
    parser.add_argument('--min_nums', default=1000, type=int, help='Min params per layer')
    parser.add_argument('--var', default=1.0, type=float, help='Target variance for generated params')
    parser.add_argument('--simulation_train', action='store_true', default=True, help='Use simulated noise instead of model training')
    parser.add_argument('--simulation_std', default=1.0, type=float, help='Std dev for simulation noise')
    parser.add_argument('--simulation_mean', default=0.0, type=float, help='Mean for simulation noise')
    parser.add_argument('--load_model', action='store_true', help='Load existing models')
    parser.add_argument('--size', default=96, type=int, help='Secret size')
    parser.add_argument('--epochs', default=5000, type=int, help='Number of epochs')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    device = torch.device(args.device)

    if args.load_model:
        secret_bits_encoder = torch.load(f"models/encoder{args.size}.pth").to(device)
        secret_bits_decoder = torch.load(f"models/decoder{args.size}.pth").to(device)
    else:
        secret_bits_encoder = SecretBitsEncoder(args.size).to(device)
        secret_bits_decoder = SecretBitsDecoder(args.size).to(device)

    optimizer_decoder = torch.optim.Adam(list(secret_bits_encoder.parameters()) + list(secret_bits_decoder.parameters()), lr=5e-5)
    criterion_decoder = torch.nn.MSELoss()

    task_model = cover_model.ResNet18().to(device)
    params = [p for p in get_model_params(task_model) if p.numel() <= 500000]
    params = torch.concatenate(params)

    bins = int(math.sqrt(len(params)))
    hist_tensor, _ = to_hist_tensor(params, bins)

    for i in range(args.epochs):
        secret_bits = get_secretbits_for_train(len(params), size=args.size).to(device)

        orignal_params = secret_bits_encoder(secret_bits)
        orignal_params = F.adaptive_max_pool1d(orignal_params.view(1, -1), len(params)).view(-1)
        orignal_params = modify_distribution(orignal_params, args.var)

        hist_params, _ = to_hist_tensor(orignal_params, bins)
        kl_divergence = F.kl_div(hist_tensor.log(), hist_params, reduction='sum')

        if args.simulation_train:
            noise_std = args.simulation_std
            inaccuracies = torch.normal(args.simulation_mean, noise_std, orignal_params.size()).to(device)
        else:
            # Full training loop logic
            model_params = orignal_params.detach().clone()
            train_loader, test_loader = get_cifar10_data()
            point = 0
            for name, m in task_model.named_modules():
                if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                    num = count_parameters(m)
                    if num > args.max_nums or num < args.min_nums:
                        continue
                    if m.bias is None:
                        m.weight = torch.nn.Parameter(model_params[point:point + num].reshape(m.weight.shape))
                    else:
                        m.bias = torch.nn.Parameter(model_params[point:point + m.bias.numel()])
                        m.weight = torch.nn.Parameter(model_params[point + m.bias.numel():point + num].reshape(m.weight.shape))
                    point += num
            
            optimizer = torch.optim.Adam(task_model.parameters(), lr=1e-4)
            train_model(task_model, train_loader, torch.nn.CrossEntropyLoss(), optimizer, num_epochs=10)
            inaccuracies = (torch.concatenate(get_model_params(task_model)).to(device) - orignal_params).detach()

        orignal_params = (orignal_params + inaccuracies).detach()
        params_to_decode = interpolate(modify_distribution(orignal_params, 1.0).view(1, 1, -1)).view(-1, 1024)
        
        outputs = secret_bits_decoder(params_to_decode)
        loss = criterion_decoder(secret_bits, outputs)
        
        optimizer_decoder.zero_grad()
        loss.backward()
        optimizer_decoder.step()

        acc = (outputs > 0.5).float().eq(secret_bits).sum().item() / secret_bits.numel()
        print(f"epoch:{i}, loss:{loss.item():.6f}, acc:{acc:.4f}, kl:{kl_divergence.item():.4f}")

    torch.save(secret_bits_encoder, f"models/encoder{args.size}.pth")
    torch.save(secret_bits_decoder, f"models/decoder{args.size}.pth")

if __name__ == "__main__":
    main()
