import torch
import torch.nn as nn
from model import CNNdeepSORT

# Instantiate the model
model = CNNdeepSORT(128, 751) # 128 is the embedding dimension, 751 is the number of classes (Market-1501 dataset)

# Move the model to the appropriate device (GPU if available)
model.to(device='cuda' if torch.cuda.is_available() else 'cpu')

# Test the model with a fake tensor, output should be of shape [B, number_classes] 
dummy_tensor = torch.randn(4, 3, 200, 200) # Shape: [B, C, H, W]
output = model(dummy_tensor)
print(output.shape)

# Define loss function, optimizer, and learning rate scheduler
loss = nn.CrossEntropyLoss() # Cross Entropy Loss is simpler, so more suitable for validation
optimizer = torch.optim.AdamW(model.parameters(), 1e-3) # AdamW is an upgraded version of Adam
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10) # StepLR reduces the learning rate by a factor of 10 every 10 epochs

# Placeholder for training function
def training():
    pass #TO BE IMPLEMENTED