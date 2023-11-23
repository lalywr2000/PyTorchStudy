import torch
import torch.nn as nn


image = torch.Tensor(1, 1, 28, 28)  # batch_size, channel, height, width
conv1 = nn.Conv2d(1, 5, 5)  # in_channels, out_channels, kernel_size
pool = nn.MaxPool2d(2)  # kernel_size


out1 = conv1(image)
out2 = pool(out1)


print(out1.shape)
print(out2.shape)
