import torch
import torch.nn as nn
import torch.nn.functional as F

#deepSORT uses a CNN to create appearance based I.D.'s, this augments SORT by allowing the model to re-identify occluded objects.
#This CNN focuses on learning spatial features.
class CNNDeepSORT(nn.Module):

    #Embedding_dim is the number of features or dimensions (independent variables)  in the representation space.
    #We chose 128 because it is a sweet spot, small enough for efficient training but large enough to capture unique features.
    #128 dimensional embeddings is a standard in object tracking/re-identification models.

    def __init__(self, embedding_dim = 128):
        super(CNNDeepSORT, self).__init__()
        '''Convolution Parameters:
        
            in_channels: 3 Channels, RGB
            out_channels: Number of feature maps the layer will produce, doubles per feature.
            kernel_size: A filter that will essentially slide along each channel and capture features using dot product.
            stride: The distance in which the kernel travels across the input channels.
            padding: Extra pixels that are added around the input image before applying convolution, this is so image details aren't lost.
        '''
        # Batch Normalization normalizes activations between the layers during training. This reduces the covariate shift
        # aka the change of the distribution of inputs from one layer to the next. Which ensures faster training.
        #x(hat) = (x - mean)/std dev
        #then y = w(x hat) + b
        #We do this to normalize the embedding values before the activation function is applied, this also reduces overfitting (by regularizing the data)

        #The first convolutional layer where the convolution math operation is performed. We are performing this in a 2D space.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        '''Global Average Pooling:
            Used instead of multiple pooling layers because it reduces the spatial dimensions of 
            the feature map to 1x1, summarizing the entire feature map into 1 vector per channel. This reduces the loss of spatial detail, which is very important
            for appearance embedding.
        '''
        #Since DeepSORT's goal is feature extraction and not classification, we only need a single fully connected layer with linear connections.
        #The linear transformation of the neurons in the last conv layer to the dense layer is y=Wx+b, which extracts the features into an
        #embedding space which is perfect for cosine similarity matching.
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=128, out_features=embedding_dim)

    #Forward propagation method for training the CNN.
    def forward(self,x):
        '''
            Convolutional Layer:
                input: Image
                output: Feature maps that highlight different patterns in the image/frame

            Batch Normalization:
                input: Feature map and number of features
                output: A feature map that will converge faster and reduce covariate shift.

            Rectified Linear Unit:
                input: The summation of all neurons in one layers' activations multplied by their weights and added their biases
                output: If input < 0, output is 0. If input > 0, output is input. This removes any negative values that do not meet a threshold.
                It also introduces non-linearity to the network, allowing for complex learning.
        '''
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.pool(x) #Downsizes the spatial dimensions of the feature map to 1x1 using Global Average Pooling
        x = torch.flatten(x,1) #Converts the multi-dimensional tensors into a 2D tensor, (batch size * features)
        x = self.fc(x) #A linear layer that matches the feature map to the desired embedding dimension (128)
        return F.normalize(x,p=2, dim=1) #Uses euclidean norm to normalize a tensor along a specific dimension

#instantiate the model
model = CNNDeepSORT()
#print(model)