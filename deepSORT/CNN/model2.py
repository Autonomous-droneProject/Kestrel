import torch
import torch.nn as nn

class CNNdeepSORT(nn.Module):

    def __init__(self, embedding_dim):
        super(CNNdeepSORT, self).__init__()
        self.embedding_dim = embedding_dim
        
        #inChannels= 3 (RGB), outChannels = embedding_dim/4 (out being # of kernels)
        chan1 = embedding_dim/4
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=chan1, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(chan1)
        self.ac1 = nn.ReLU(True)
                
        chan2 = embedding_dim/2
        self.conv2 = nn.Conv2d(in_channels=chan1, out_channels=chan2, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(chan2)
        self.ac2 = nn.ReLU(True)
        
        self.conv3 = nn.Conv3d(in_channels=chan2, out_channels=embedding_dim, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(embedding_dim)
        self.ac3 = nn.ReLU(True)

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.ll = nn.Linear(in_features = embedding_dim, out_features=embedding_dim)

    


        