import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNdeepSORT(nn.Module):

    def __init__(self, embedding_dim):
        super(CNNdeepSORT, self).__init__()
        
        
        chan1 = embedding_dim//4
        chan2 = embedding_dim//2
        self.convolution = nn.Sequential(
            #Input Dimensions: [B, 3, H, W]
            #inChannels= 3 (RGB), outChannels = embedding_dim/4 (out being # of kernels)
            
            nn.Conv2d(in_channels=3, out_channels=chan1, kernel_size=3),
            nn.BatchNorm2d(chan1),
            nn.ReLU(True),
                    
            
            nn.Conv2d(in_channels=chan1, out_channels=chan2, kernel_size=3),
            nn.BatchNorm2d(chan2),
            nn.ReLU(True),
            
            nn.Conv2d(in_channels=chan2, out_channels=embedding_dim, kernel_size=3),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(True),

            
            nn.AdaptiveAvgPool2d((1,1))
            #After pooling: [B, 128, 1, 1]
        )

        
        self.linearLayer = nn.Linear(in_features = embedding_dim, out_features=embedding_dim)
        
    def forward(self, inputTensor):
        
        output = self.convolution(inputTensor)
        output = torch.flatten(output,1)
        #After Flatenning: [B, 128] -> person's appearance vector 
        output = self.linearLayer(output)
        
        return output
    


        