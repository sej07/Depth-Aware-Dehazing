import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class PixelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        return x * y


class FABlock(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        self.ca = ChannelAttention(channels)
        self.pa = PixelAttention(channels)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.ca(out)
        out = self.pa(out)
        
        return out + x


class FAGroup(nn.Module):
    def __init__(self, channels, num_blocks):
        super().__init__()
        
        self.blocks = nn.Sequential(*[FABlock(channels) for _ in range(num_blocks)])
        
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        out = self.blocks(x)
        out = self.conv(out)
        return out + x


class FFANet(nn.Module):
    
    def __init__(self, in_channels=3, channels=64, num_groups=3, num_blocks=6):
        super().__init__()
        
        self.in_channels = in_channels
        
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        
        self.groups = nn.ModuleList([
            FAGroup(channels, num_blocks) for _ in range(num_groups)
        ])

        self.conv_fusion = nn.Conv2d(channels * num_groups, channels, kernel_size=1)
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        if self.in_channels == 3:
            residual = x
        else:
            residual = x[:, :3, :, :]
        f = self.conv_in(x)
        
        group_outputs = []
        for group in self.groups:
            f = group(f)
            group_outputs.append(f)
            
        fused = torch.cat(group_outputs, dim=1)
        fused = self.conv_fusion(fused)
        
        out = self.conv_out(fused)
        out = out + residual
        
        out = torch.clamp(out, 0, 1)
        
        return out
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)