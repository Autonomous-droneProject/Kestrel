import torch
from model import CNNdeepSORT

model = CNNdeepSORT(128, 128)
fake_tensor = torch.randn(4, 3, 200, 200)
output = model(fake_tensor)
print(output.shape)