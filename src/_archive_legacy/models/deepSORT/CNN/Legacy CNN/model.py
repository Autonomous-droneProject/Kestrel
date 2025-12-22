import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNdeepSORT(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        
        self.convolution = nn.Sequential(
            #Input Dimensions: [B, 3, H, W]
            #inChannels= 3 (RGB), outChannels = embedding_dim/4 (out being # of kernels)
            
            # Stage 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), # without padding, convolution shrinks images quickly, potentially losing info @ edges.
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2), #downsample using maxmimum values in kernel -> H/2 x W/2

            # Stage 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2), # H/2 x W/2

            # Stage 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=embedding_dim, kernel_size=3, padding=1), nn.BatchNorm2d(embedding_dim), nn.ReLU(True),
            
            #Final reduction later into 1x1 embedding: [B, embedding_dim, 1, 1]
            nn.AdaptiveAvgPool2d((1,1)) 
        )

        # Fully connected classifier for Re-ID training
        #self.classifier = nn.Linear(in_features = embedding_dim, out_features=num_classes)
        
    def forward(self, inputTensor):
        
        output = self.convolution(inputTensor)  # CNN encoder block -> [B, emb_dim, 1, 1]
        output = torch.flatten(output,1)        # [B, emb_dim] -> person's appearance vector 
        
        output = self.classifier(output)   # For use as classifier -> [B, num_classes]
        # else: For use as feature extractor        
        return output