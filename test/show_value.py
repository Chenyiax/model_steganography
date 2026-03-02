import torch
import matplotlib.pyplot as plt

params_with_secret = torch.load("../data/params_init_with_secret.pth")
params_without_secret = torch.load("../data/params_init_without_secret.pth")

params_without_secret = params_without_secret[0]
params_with_secret = params_with_secret[0]

params_without_secret, _ = torch.sort(params_without_secret)
params_with_secret, _ = torch.sort(params_with_secret)

plt.plot(params_without_secret, label="cover")
plt.plot(params_with_secret, label="stego")
plt.legend()

plt.show()