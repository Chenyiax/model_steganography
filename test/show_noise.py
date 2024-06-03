import math

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils import to_hist_tensor

noise = torch.load("../data/inacc.pth")
noise = torch.concatenate(noise)
normal = torch.normal(torch.mean(noise).item(), torch.std(noise).item(), size=noise.size())

bins = int(math.sqrt(len(noise)))

hist_tensor1, bin_center1 = to_hist_tensor(noise, bins)
hist_tensor2, bin_center2 = to_hist_tensor(normal, bins)

kl_divergence = F.kl_div(hist_tensor1.log(), hist_tensor2, reduction='sum')

plt.plot(bin_center1, hist_tensor1, label='model train noise')
plt.plot(bin_center2, hist_tensor2, label='normal noise')
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.title('model train noise and normal noise')
plt.legend()
plt.show()