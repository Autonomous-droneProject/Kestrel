import torch
import torch.nn as nn
from model import CNNdeepSORT
from torch.utils.data import DataLoader
import dataset
from tqdm import tqdm
import time
from torchvision import transforms

# Instantiate the model
model = CNNdeepSORT(128, 751) # 128 is the embedding dimension, 751 is the number of classes (Market-1501 dataset)

# Move the model to the appropriate device (GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

'''
# Test the model with a fake tensor, output should be of shape [B, number_classes] 
dummy_tensor = torch.randn(4, 3, 200, 200).to(device) # Shape: [B, C, H, W]
output = model(dummy_tensor)
print(output.shape)
'''

#HYPER PARAMETERS
LEARNING_RATE = 1e-3
epochs = 50


# Define loss function, optimizer, and learning rate scheduler
loss_function = nn.CrossEntropyLoss() # Cross Entropy Loss is simpler, so more suitable for validation
optimizer = torch.optim.AdamW(model.parameters(), LEARNING_RATE) # AdamW is an upgraded version of Adam
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10) # StepLR reduces the learning rate by a factor of 10 every 10 epochs


Market1501_train_file_path = r'C:\Users\adamm\PROJECTS\Market 1501\Market-1501-v15.09.15\bounding_box_train'
transform = transforms.Compose([transforms.ToTensor()])

data = dataset.Market1501(Market1501_train_file_path, transform)
batch_size = 64 #Number of samples to be used for training at once, should be determined based on hardware specifications. 32 and 64 are common.


#num_workers is the number of multi-threaded processes run for training your model at the same time
#We found that num_workers = 4 * num_cores (CPU)
training_dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

#Training Function
def training(model, training_dataloader, loss_function, optimizer):
    progress_bar = tqdm(training_dataloader, "Training Progress...")
    model.train()
    total_training_loss = 0
    
    for (image, label) in progress_bar:
        image = image.to(device)
        label = label.to(device)
        
        prediction = model(image) #Forward propagation
        current_loss = loss_function(prediction, label) #Find out how wrong we were
        
        current_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_training_loss += current_loss.item()
        
        progress_bar.set_postfix(current_loss=current_loss.item())
    
    return total_training_loss


#Training Loop
for epoch in range(epochs):
    start_time = time.time()
    total_training_loss = training(model, training_dataloader, loss_function, optimizer)
    print(f"Epoch: {epoch+1}/{epochs}")  
    print(f"Total Training Loss = {total_training_loss}")
    elapsed_time = time.time() - start_time
    print(f"Time per epoch: {elapsed_time}s")
        
