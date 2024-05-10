import torch

secret_bits_encoder = torch.load(f"encoder3_128.pth")
secret_bits_decoder = torch.load(f"decoder3_128.pth")

torch.save(secret_bits_encoder, "data/encoder.pth")
torch.save(secret_bits_decoder, "data/decoder.pth")