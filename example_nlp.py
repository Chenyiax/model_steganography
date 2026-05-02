"""
Filename: example_nlp.py
Description: An NLP example where a model embeds secret information and then extracts it.
"""
import argparse
import copy
import torch
import stego_model
from utils.init_function import init_nlp
from model_steganography import ModelSteganography
from utils.get_data import get_sst2_data
from utils.test import test_model
from utils.train import train_model

def main():
    parser = argparse.ArgumentParser(description="NLP Steganography Example")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--target_var", type=float, default=1e-4, help="Target variance for steganography")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer"], help="Model architecture")
    parser.add_argument("--save", action="store_true", help="Save the steganographic model")
    
    args = parser.parse_args()

    init_func = init_nlp
    # Object-oriented programming, generate a model steganography class
    ms = ModelSteganography(init_func, target_var=args.target_var)

    train_loader, test_loader, vocab_size, vocab_len = get_sst2_data()

    if args.model == "lstm":
        task_model = cover_model.LSTM(vocab_size)
    else:
        task_model = cover_model.TransformerClassifier(vocab_size, vocab_len)
        
    print(task_model)

    # Generate and embed secret information
    secret_bits, secret_bits_bch = ms.encode(task_model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(task_model.parameters(), lr=args.lr)

    # Steganographic model training
    train_model(task_model, train_loader, criterion, optimizer, num_epochs=args.epochs)
    test_model(task_model, test_loader, criterion)

    # Save model if requested
    if args.save:
        torch.save(task_model, f"models/{task_model.__class__.__name__}_with_secret.pth")
        print(f"Model saved to models/{task_model.__class__.__name__}_with_secret.pth")

    # Extract secret information
    outputs_secrets, outputs_secrets_bch = ms.decode(task_model)

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
