import torch
import torch.optim as optim
import torch.nn as nn
from CNN_Data_Processing import train_loader
from CNN_model import CNNDeepSORT  # Assuming this is your CNN model
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Triplet Loss
criterion = nn.TripletMarginLoss(margin=1.0, p=2)

# Initialize Model
model = CNNDeepSORT().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (anchor, positive, negative) in enumerate(train_loader):
            if anchor.shape[0] == 0:
                continue  # Skip empty batches

            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
            optimizer.zero_grad()

            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)

            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {running_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "CNNDeepSORT_final.pth")
    print("Training complete! Model saved to CNNDeepSORT_final.pth")

train(model, train_loader, criterion, optimizer, num_epochs=10)
