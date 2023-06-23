import torch
import torch.nn as nn
import torch.nn.functional as F

class BNNet(nn.Module):
    def __init__(self):
        super(BNNet, self).__init__()

        # C1 Convolutional Layer
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)  
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()

        # C2 Convolutional Layer
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1) 
        self.bn2 = nn.BatchNorm2d(12)
        self.relu2 = nn.ReLU()

        # C3 Convolutional Layer 
        self.conv3 = nn.Conv2d(12, 12, 1)
        self.bn3 = nn.BatchNorm2d(12)
        self.relu3 = nn.ReLU()

        # P1 Pooling Layer
        self.pool1 = nn.MaxPool2d(2, 2)

        # C4 Convolutional Layer
        self.conv4 = nn.Conv2d(12, 24, 3, padding=1) 
        self.bn4 = nn.BatchNorm2d(24)
        self.relu4 = nn.ReLU()

        # C5 Convolutional Layer
        self.conv5 = nn.Conv2d(24, 24, 3, padding=1) 
        self.bn5 = nn.BatchNorm2d(24)
        self.relu5 = nn.ReLU()

        # C6 Convolutional Layer 
        self.conv6 = nn.Conv2d(24, 24, 1)
        self.bn6 = nn.BatchNorm2d(24)
        self.relu6 = nn.ReLU()

        # P2 Pooling Layer
        self.pool2 = nn.MaxPool2d(2, 2)

        # C7 Convolutional Layer
        self.conv7 = nn.Conv2d(24, 48, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(48)
        self.relu7 = nn.ReLU()

        # C8 Convolutional Layer
        self.conv8 = nn.Conv2d(48, 48, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(48)
        self.relu8 = nn.ReLU()

        # C9 Convolutional Layer
        self.conv9 = nn.Conv2d(48, 48, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(48)
        self.relu9 = nn.ReLU()

        # GAP layer
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # C10 Convolutional Layer
        self.conv10 = nn.Conv2d(48, 10, 1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.pool1(x)
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.pool2(x)
        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.relu8(self.bn8(self.conv8(x)))
        x = self.relu9(self.bn9(self.conv9(x)))
        x = self.avgpool(x)
        x = self.conv10(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)

class GNNet(nn.Module):
    def __init__(self):
        super(GNNet, self).__init__()
        num_groups = 3  # Please note: For group norm, num_groups argument can be tuned as per needs

        # C1 Convolutional Layer
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)  
        self.gn1 = nn.GroupNorm(num_groups, 6)
        self.relu1 = nn.ReLU()

        # C2 Convolutional Layer
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, 12)
        self.relu2 = nn.ReLU()

        # C3 Convolutional Layer 
        self.conv3 = nn.Conv2d(12, 12, 1)
        self.gn3 = nn.GroupNorm(num_groups, 12)
        self.relu3 = nn.ReLU()

        # P1 Pooling Layer
        self.pool1 = nn.MaxPool2d(2, 2)

        # C4 Convolutional Layer
        self.conv4 = nn.Conv2d(12, 24, 3, padding=1)
        self.gn4 = nn.GroupNorm(num_groups, 24)
        self.relu4 = nn.ReLU()

        # C5 Convolutional Layer
        self.conv5 = nn.Conv2d(24, 24, 3, padding=1)
        self.gn5 = nn.GroupNorm(num_groups, 24)
        self.relu5 = nn.ReLU()

        # C6 Convolutional Layer 
        self.conv6 = nn.Conv2d(24, 24, 1)
        self.gn6 = nn.GroupNorm(num_groups, 24)
        self.relu6 = nn.ReLU()

        # P2 Pooling Layer
        self.pool2 = nn.MaxPool2d(2, 2)

        # C7 Convolutional Layer
        self.conv7 = nn.Conv2d(24, 48, 3, padding=1)
        self.gn7 = nn.GroupNorm(num_groups, 48)
        self.relu7 = nn.ReLU()

        # C8 Convolutional Layer
        self.conv8 = nn.Conv2d(48, 48, 3, padding=1)
        self.gn8 = nn.GroupNorm(num_groups, 48)
        self.relu8 = nn.ReLU()

        # C9 Convolutional Layer
        self.conv9 = nn.Conv2d(48, 48, 3, padding=1)
        self.gn9 = nn.GroupNorm(num_groups, 48)
        self.relu9 = nn.ReLU()

        # GAP layer
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # C10 Convolutional Layer
        self.conv10 = nn.Conv2d(48, 10, 1)

    def forward(self, x):
        x = self.relu1(self.gn1(self.conv1(x)))
        x = self.relu2(self.gn2(self.conv2(x)))
        x = self.relu3(self.gn3(self.conv3(x)))
        x = self.pool1(x)
        x = self.relu4(self.gn4(self.conv4(x)))
        x = self.relu5(self.gn5(self.conv5(x)))
        x = self.relu6(self.gn6(self.conv6(x)))
        x = self.pool2(x)
        x = self.relu7(self.gn7(self.conv7(x)))
        x = self.relu8(self.gn8(self.conv8(x)))
        x = self.relu9(self.gn9(self.conv9(x)))
        x = self.avgpool(x)
        x = self.conv10(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)

class LNNet(nn.Module):
    def __init__(self):
        super(LNNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.ln1 = nn.LayerNorm([6, 32, 32])
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        self.ln2 = nn.LayerNorm([12, 32, 32])
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(12, 12, 1)
        self.ln3 = nn.LayerNorm([12, 32, 32])
        self.relu3 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(12, 24, 3, padding=1)
        self.ln4 = nn.LayerNorm([24, 16, 16])
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(24, 24, 3, padding=1)
        self.ln5 = nn.LayerNorm([24, 16, 16])
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(24, 24, 1)
        self.ln6 = nn.LayerNorm([24, 16, 16])
        self.relu6 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(24, 48, 3, padding=1)
        self.ln7 = nn.LayerNorm([48, 8, 8])
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(48, 48, 3, padding=1)
        self.ln8 = nn.LayerNorm([48, 8, 8])
        self.relu8 = nn.ReLU()

        self.conv9 = nn.Conv2d(48, 48, 3, padding=1)
        self.ln9 = nn.LayerNorm([48, 8, 8])
        self.relu9 = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv10 = nn.Conv2d(48, 10, 1)

    def forward(self, x):
        x = self.relu1(self.ln1(self.conv1(x)))
        x = self.relu2(self.ln2(self.conv2(x)))
        x = self.relu3(self.ln3(self.conv3(x)))
        x = self.pool1(x)
        x = self.relu4(self.ln4(self.conv4(x)))
        x = self.relu5(self.ln5(self.conv5(x)))
        x = self.relu6(self.ln6(self.conv6(x)))
        x = self.pool2(x)
        x = self.relu7(self.ln7(self.conv7(x)))
        x = self.relu8(self.ln8(self.conv8(x)))
        x = self.relu9(self.ln9(self.conv9(x)))
        x = self.avgpool(x)
        x = self.conv10(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)


