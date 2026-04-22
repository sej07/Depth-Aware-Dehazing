import torch
import torch.nn as nn


class AODNet(nn.Module):
    """
    Key idea: Reformulate atmospheric scattering model to directly estimate K(x)
    J(x) = (I(x) - A) / K(x) + A
    """
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.in_channels = in_channels
        
        # Layer 1: 1×1 conv
        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Layer 2: 3×3 conv
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Layer 3: 5×5 conv
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Layer 4: 7×7 conv
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, padding=3)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Layer 5: 3×3 conv
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, padding=1)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        if self.in_channels == 3:
            hazy = x
        else:
            hazy = x[:, :3, :, :]
        
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(x1))
        
        cat1 = torch.cat([x1, x2], dim=1) 
        
        x3 = self.relu3(self.conv3(cat1))
        
        
        cat2 = torch.cat([x2, x3], dim=1)
        
        x4 = self.relu4(self.conv4(cat2))
        
        cat3 = torch.cat([x1, x2, x3, x4], dim=1) 
        k = self.conv5(cat3) 
        k = torch.relu(k) + 0.0001  
        output = k * hazy - k + 1
        output = torch.clamp(output, 0, 1)
        
        return output
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)