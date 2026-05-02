"""
Filename: adversarial _training.py
Description: Adversarial training script to train the generator and discriminator for steganography.
"""
import argparse
import random
import math
import torch
import torch.nn.functional as F
import stego_model
from model import Discriminator
from utils.util import get_model_params, to_hist_tensor, compute_accuracy, get_secretbits_for_train, modify_distribution

def main():
    parser = argparse.ArgumentParser(description="GAN Training for Steganography")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--size", type=int, default=128, help="Secret size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    
    args = parser.parse_args()

    device = torch.device(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion_decoder = torch.nn.MSELoss()

    secret_bits_encoder = torch.load(f"models/encoder{args.size}.pth").to(device)
    secret_bits_decoder = torch.load(f"models/decoder{args.size}.pth").to(device)
    discriminator = Discriminator().to(device)

    parameters = list(secret_bits_encoder.parameters()) + list(secret_bits_decoder.parameters())
    optimizer_encoder = torch.optim.Adam(parameters, lr=args.lr)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    for epoch_i in range(args.epochs):
        print(f"############################################\nepoch: {epoch_i}")
        task_model = cover_model.ResNet18().to(device)

        orignal_params_list = get_model_params(task_model)
        params_without_secret = random.choice(orignal_params_list).to(device)

        var = torch.var(params_without_secret).item()
        secret_bit = get_secretbits_for_train(len(params_without_secret), size=args.size).to(device)
        
        params_with_secret = secret_bits_encoder(secret_bit)
        params_with_secret = F.adaptive_max_pool1d(params_with_secret.view(1, -1), len(params_without_secret)).view(-1)
        params_with_secret = modify_distribution(params_with_secret, var)

        random_val = random.randrange(2)
        noise_mean = 0.0027 if random_val == 0 else -0.0027
        noise = torch.normal(noise_mean, 0.04, params_with_secret.size()).to(device)

        params_without_secret += noise
        params_with_secret += noise

        params_to_decode = params_with_secret.clone()

        nums = params_with_secret.size(-1)
        batch = nums // 1024 + 1
        target = batch * 1024
        scale_factor = target / nums + 1e-9

        params_with_secret = F.interpolate(params_with_secret.view(1, 1, -1), scale_factor=scale_factor, mode='linear', align_corners=False).view(batch, 1024).detach()
        params_without_secret = F.interpolate(params_without_secret.view(1, 1, -1), scale_factor=scale_factor, mode='linear', align_corners=False).view(batch, 1024).detach()
        params_to_decode = F.interpolate(params_to_decode.view(1, 1, -1), scale_factor=scale_factor, mode='linear', align_corners=False).view(batch, 1024)

        bins = int(math.sqrt(len(params_with_secret.view(-1))))
        hist_tensor1, _ = to_hist_tensor(params_with_secret.view(-1), bins=bins)
        hist_tensor2, _ = to_hist_tensor(params_without_secret.view(-1), bins=bins)

        kl_divergence = F.kl_div(hist_tensor1.log(), hist_tensor2, reduction='sum')
        print(f"KL Divergence: {kl_divergence.item()}")

        # Train Discriminator
        optimizer_discriminator.zero_grad()
        dx = discriminator(params_with_secret.detach())
        dg = discriminator(params_without_secret.detach())
        loss_discriminator = criterion(dx, torch.ones_like(dx)) + criterion(dg, torch.zeros_like(dg))
        loss_discriminator.backward()
        optimizer_discriminator.step()

        acc_real = compute_accuracy(dx, torch.ones_like(dx[:, 0]))
        acc_fake = compute_accuracy(dg, torch.zeros_like(dg[:, 0]))
        print(f"Discriminator Acc: {(acc_real + acc_fake) / 2}, Loss: {loss_discriminator.item()}")

        # Train Encoder/Decoder
        optimizer_encoder.zero_grad()
        df = discriminator(params_to_decode)
        loss_encoder = criterion(df, torch.ones_like(df))
        ga = secret_bits_decoder(params_to_decode)
        loss_decoder = criterion_decoder(ga, secret_bit)
        loss = loss_decoder + loss_encoder
        
        acc = (ga > 0.5).float().eq(secret_bit).sum().item() / (secret_bit.size(0) * args.size)
        loss.backward()
        optimizer_encoder.step()
        
        print(f"Encoder Loss: {loss_encoder.item()}, Decoder Acc: {acc}")

    torch.save(discriminator, f"models/discriminator.pth")
    torch.save(secret_bits_encoder, f"models/encoder{args.size}.pth")
    torch.save(secret_bits_decoder, f"models/decoder{args.size}.pth")

if __name__ == "__main__":
    main()
