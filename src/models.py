import torch
import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn as nn

class CNN3D(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.3):
        super(CNN3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 96, kernel_size=(7, 7, 7), stride=2, padding=3),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2),
            
            nn.Conv3d(96, 256, kernel_size=(5, 5, 5), padding=2),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2),
            
            nn.Conv3d(256, 384, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(384),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(384, 384, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(384),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(384, 256, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        # x is of shape (batch_size, channels, depth, height, width)
        # Process each slice independently through the 2D model
        batch_size, channels, depth, height, width = x.shape
        
        features = []
        for i in range(depth):
            slice = x[:, :, i, :, :]  # Extracting each 2D slice from depth dimension
            
            # Convert 1-channel to 3-channel by repeating the grayscale channel 3 times
            slice = slice.repeat(1, 3, 1, 1)
            
            slice = self.model.features(slice)  # Pass through AlexNet feature extractor
            slice = self.gap(slice).view(slice.size(0), -1)
            features.append(slice)
        
        features = torch.stack(features, dim=0)  # Stack features from all slices
        x = torch.max(features, 0)[0]  # Aggregate features across slices using max pooling
        x = self.dropout(x)
        x = self.classifier(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT  # Use the most up-to-date weights
        self.model = models.resnet50(weights=weights)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Adjust for single-channel input
        self.model.fc = nn.Identity()  # Remove the final fully connected layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(2048, 1)

    def forward(self, x):
        batch_size, channels, depth, height, width = x.shape
        
        features = []
        for i in range(depth):
            slice = x[:, :, i, :, :]  # Extracting each 2D slice
            slice = self.model(slice)  # Pass through ResNet feature extractor
            if len(slice.shape) == 4:  # Ensure slice has correct shape for GAP
                slice = self.gap(slice).view(slice.size(0), -1)
            features.append(slice)
        
        features = torch.stack(features, dim=0)  # Stack features from all slices
        features = torch.max(features, dim=0)[0]  # Aggregate features across slices using max pooling
        features = self.dropout(features)
        x = self.classifier(features)
        return x
    
from torch.utils.checkpoint import checkpoint
class VisionTransformer(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        num_features = self.model.heads.head.in_features 
        self.model.heads = nn.Identity()  # Remove the default classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_features, 1)  # Add a new classification head

    def forward(self, x):
        batch_size, channels, depth, height, width = x.shape
        
        features = []
        for i in range(depth):
            slice = x[:, :, i, :, :]  # Extracting each 2D slice
            slice = slice.repeat(1, 3, 1, 1)  # Convert to 3-channel
            slice.requires_grad_()  # Ensure requires_grad is set to True
            slice = checkpoint(self.model, slice, use_reentrant=True)  # Use checkpointing
            features.append(slice)
        
        features = torch.stack(features, dim=0)  # Stack features from all slices
        x = torch.max(features, dim=0)[0]  # Aggregate features across slices using max pooling
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
class SwinTransformerV1(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        num_features = self.model.head.in_features
        self.model.head = nn.Identity()  # Remove the classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_features, 1)

    def forward(self, x):
        batch_size, channels, depth, height, width = x.shape
        
        features = []
        for i in range(depth):
            slice = x[:, :, i, :, :]  # Extracting each 2D slice
            slice = slice.repeat(1, 3, 1, 1)  # Convert to 3-channel
            slice.requires_grad_()  # Ensure requires_grad is set to True
            slice = checkpoint(self.model, slice, use_reentrant=True)  # Use checkpointing
            features.append(slice)
        
        features = torch.stack(features, dim=0)  # Stack features from all slices
        features = torch.max(features, dim=0)[0]  # Aggregate features across slices using max pooling
        features = self.dropout(features)
        x = self.classifier(features)
        return x

from torchvision.transforms import Resize #can remove this if not resizing
class SwinTransformerV2(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1)
        num_features = self.model.head.in_features
        self.model.head = nn.Identity() # Remove the classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_features, 1)
        #self.resize = Resize((256, 256))

    def forward(self, x):
        batch_size, channels, depth, height, width = x.shape
        
        features = []
        for i in range(depth):
            slice = x[:, :, i, :, :]  # Extract each 2D slice
            #slice = self.resize(slice)  # Resize the slice to 256x256 if needed
            slice = slice.repeat(1, 3, 1, 1)  # Convert to 3-channel if it's single-channel
            slice.requires_grad_()  # Ensure requires_grad is set to True
            slice = checkpoint(self.model, slice, use_reentrant=True)  # Use checkpointing
            features.append(slice)
        
        features = torch.stack(features, dim=0)  # Stack features from all slices
        features = torch.max(features, dim=0)[0]  # Max pooling along the depth dimension
        features = self.dropout(features)
        x = self.classifier(features)
        
        return x


class DenseNet(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Adjust for single-channel input
        self.model.classifier = nn.Identity()  # Remove the final classifier layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(1024, 1)  # Adjust according to the number of features in DenseNet

    def forward(self, x):
        batch_size, channels, depth, height, width = x.shape
        
        features = []
        for i in range(depth):
            slice = x[:, :, i, :, :]  # Extracting each 2D slice
            slice = self.model.features(slice)  # Pass through DenseNet feature extractor
            slice = self.gap(slice).view(slice.size(0), -1)
            features.append(slice)
        
        features = torch.stack(features, dim=0)  # Stack features from all slices
        features = torch.max(features, dim=0)[0]  # Aggregate features across slices using max pooling
        features = self.dropout(features)
        x = self.classifier(features)
        return x

class EfficientNet(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)  # Adjust for single-channel input
        self.model.classifier = nn.Identity()  # Remove the final classifier layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(1280, 1)  # Adjust according to the number of features in EfficientNet

    def forward(self, x):
        batch_size, channels, depth, height, width = x.shape

        features = []
        for i in range(depth):
            slice = x[:, :, i, :, :]  # Extracting each 2D slice
            slice = self.model.features(slice)  # Pass through EfficientNet feature extractor
            slice = self.gap(slice).view(slice.size(0), -1)
            features.append(slice)

        features = torch.stack(features, dim=0)  # Stack features from all slices
        features = torch.max(features, dim=0)[0]  # Aggregate features across slices using max pooling
        features = self.dropout(features)
        x = self.classifier(features)
        return x

class ResNet34(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT  # Use the most up-to-date weights
        self.model = models.resnet34(weights=weights)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Adjust for single-channel input
        self.model.fc = nn.Identity()  # Remove the final fully connected layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        batch_size, channels, depth, height, width = x.shape

        features = []
        for i in range(depth):
            slice = x[:, :, i, :, :]  # Extracting each 2D slice
            slice = self.model(slice)  # Pass through ResNet feature extractor
            if len(slice.shape) == 4:  # Ensure slice has correct shape for GAP
                slice = self.gap(slice).view(slice.size(0), -1)
            features.append(slice)

        features = torch.stack(features, dim=0)  # Stack features from all slices
        features = torch.max(features, dim=0)[0]  # Aggregate features across slices using max pooling
        features = self.dropout(features)
        x = self.classifier(features)
        return x

