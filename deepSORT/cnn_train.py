import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from CNNDeepSort import CNNDeepSORT  # Importing model
from cnn_data_preparation import load_annotations, TrackingDataset, collate_fn  # Import dataset functions

#Define file paths
video_path = r"C:\Users\adamm\PycharmProjects\Kestrel\deepSORT\train\MOT17-02\img1"
ini_file = r"C:\Users\adamm\PycharmProjects\Kestrel\deepSORT\train\MOT17-02\seqinfo.ini"

#Load annotations
annotations = load_annotations(ini_file)

#Define data transformations (resize, normalize)
transform = transforms.Compose([
  transforms.Resize((224, 224)), #Resize to fixed dimensions
    transforms.ToTensor(), #Convert to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #ImageNet normalization
])

tracking_dataset = TrackingDataset(video_path, annotations, transform=transform)
train_loader = DataLoader(tracking_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

#initialize the CNN
device = torch.device("cpu")
model = CNNDeepSORT().to(device) #Instantiate the model to the torch device

#Define the optimizer and loss functions
optimizer = optim.Adam(model.parameters(), lr = 0.001)
loss_function = torch.nn.TripletMarginLoss(margin=1.0, p=2)

#Training loop!
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        if batch is None:
            continue #Skip any empty batches

        anchor, positive, negative = batch
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        #Forward propagation
        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)

        #Compute the loss using Triplet Loss
        loss = loss_function(anchor_embedding, positive_embedding, negative_embedding)
        total_loss += loss.item()

        #Backward propagation and optimizer!
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #Printing out the loss per epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader)}")

#Save the trained model!!!
torch.save(model.state_dict(), "deepsort_cnn.pth")
print("Training complete, model has been saved")


