"""
Filename: example_cv.py
Description: A CV example where a model embeds secret information and then extracts it.
"""

import argparse
import random
import torch
import torch.nn as nn
import stego_model
from utils.init_function import init_vit, init_resnet, init_alexnet, init_densenet, init_vgg
from utils.get_data import get_fashionmnist_data, get_cifar10_data, get_mnist_data
from utils.train import train_model_with_extract
from utils.test import test_model
from model_steganography import ModelSteganography
from utils.util import get_model_params, count_parameters

def main():
    parser = argparse.ArgumentParser(description="CV Steganography Example")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--target_var", type=float, default=2e-4, help="Target variance for steganography")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist","fashionmnist", "cifar10"], help="Dataset to use")
    parser.add_argument("--model", type=str, default="vit", choices=["alexnet", "resnet", "densenet", "vgg16", "vit"], help="Model architecture")
    parser.add_argument("--save", action="store_true", help="Save the steganographic model")
    parser.add_argument("--fast", action="store_true", help="Use fast noise-based injection instead of training")
    
    args = parser.parse_args()

    # Load dataset
    if args.dataset == "fashionmnist":
        train_loader, test_loader = get_fashionmnist_data()
    elif args.dataset == "mnist":
        train_loader, test_loader = get_mnist_data()
    else:
        train_loader, test_loader = get_cifar10_data()

    # Initialize model and init_func
    model_map = {
        "alexnet": (cover_model.AlexNet, init_alexnet),
        "resnet": (cover_model.ResNet18, init_resnet),
        "densenet": (cover_model.DenseNet, init_densenet),
        "vgg16": (cover_model.Vgg16, init_vgg),
        "vit": (cover_model.VisionTransformer, init_vit),
    }
    
    model_class, init_func = model_map[args.model]
    stego_model = model_class()
    print(stego_model)
    
    # Object-oriented programming, generate a model steganography class
    ms = ModelSteganography(init_func, target_var=args.target_var)
    
    # Perform secret-containing initialization on the carrier model
    secret_bits, secret_bits_bch = ms.encode(stego_model)
    print(f"Secret bits: {secret_bits.numel()}, BCH bits: {secret_bits_bch.numel()}")

    if args.fast:
        # Fast noise-based injection
        params = get_model_params(stego_model)
        i = 0
        for name, m in stego_model.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                params_num = count_parameters(m)
                if params_num > 5000000 or params_num < 1000:
                    continue

                random_integer = random.randrange(2)
                if random_integer == 0:
                    noise = torch.normal(0.0027, 0.03, params[i].size())
                else:
                    noise = torch.normal(-0.0027, 0.03, params[i].size())

                params[i] = params[i] + noise
                if m.bias is None:
                    m.weight = nn.Parameter(params[i].reshape(m.weight.shape))
                else:
                    m.bias = nn.Parameter(params[i][:m.bias.numel()])
                    m.weight = nn.Parameter(params[i][m.bias.numel():].reshape(m.weight.shape))
                i += 1
    else:
        # Train and test the steganographic model
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(stego_model.parameters(), lr=args.lr)
        train_model_with_extract(stego_model, train_loader, criterion, optimizer, ms, secret_bits, secret_bits_bch, num_epochs=args.epochs)
        test_model(stego_model, test_loader, criterion)
    
    # Save model if requested
    if args.save:
        torch.save(stego_model, f"models/{stego_model.__class__.__name__}_with_secret.pth")
        print(f"Model saved to models/{stego_model.__class__.__name__}_with_secret.pth")

    # Extract secret information
    outputs_secrets, outputs_secrets_bch = ms.decode(stego_model)

    correct = (outputs_secrets == secret_bits).sum().item()
    accuracy = correct / outputs_secrets.numel()
    print("Extraction Accuracy of Secret Information:", accuracy)

    correct = (outputs_secrets_bch == secret_bits_bch).sum().item()
    accuracy = correct / outputs_secrets_bch.numel()
    print("Extraction Accuracy of Secret Information after BCH:", accuracy)
    print("secret numel:", outputs_secrets.numel(), "bits")
    print("bch secret numel:", outputs_secrets_bch.numel(), "bits")

if __name__ == "__main__":
    main()
