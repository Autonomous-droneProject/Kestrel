import torch
import torch.nn as nn
from model import CNNdeepSORT

model = CNNdeepSORT(128, 751)
fake_tensor = torch.randn(4, 3, 200, 200)
output = model(fake_tensor)
print(output.shape)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), 0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)

def training():
    pass