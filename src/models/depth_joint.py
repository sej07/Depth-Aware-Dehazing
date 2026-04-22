import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        f = self.conv(x)
        p = self.pool(f)
        return f, p


class DecoderBlock(nn.Module):
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class JointDehazeDepthNet(nn.Module):
    
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        self.in_channels = in_channels
        c = base_channels
        
        self.enc1 = EncoderBlock(in_channels, c) 
        self.enc2 = EncoderBlock(c, c * 2)
        self.enc3 = EncoderBlock(c * 2, c * 4) 
        self.enc4 = EncoderBlock(c * 4, c * 8)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c * 8, c * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 16, c * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.dehaze_dec4 = DecoderBlock(c * 16, c * 8, c * 8)
        self.dehaze_dec3 = DecoderBlock(c * 8, c * 4, c * 4)
        self.dehaze_dec2 = DecoderBlock(c * 4, c * 2, c * 2)
        self.dehaze_dec1 = DecoderBlock(c * 2, c, c)
        self.dehaze_out = nn.Conv2d(c, 3, kernel_size=1)
        
        self.depth_dec4 = DecoderBlock(c * 16, c * 8, c * 8)
        self.depth_dec3 = DecoderBlock(c * 8, c * 4, c * 4)
        self.depth_dec2 = DecoderBlock(c * 4, c * 2, c * 2)
        self.depth_dec1 = DecoderBlock(c * 2, c, c)
        self.depth_out = nn.Conv2d(c, 1, kernel_size=1)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        residual = x
        
        f1, p1 = self.enc1(x) 
        f2, p2 = self.enc2(p1)
        f3, p3 = self.enc3(p2)
        f4, p4 = self.enc4(p3) 

        bn = self.bottleneck(p4)
        
        dh = self.dehaze_dec4(bn, f4)
        dh = self.dehaze_dec3(dh, f3)
        dh = self.dehaze_dec2(dh, f2)
        dh = self.dehaze_dec1(dh, f1)
        dehazed = self.dehaze_out(dh)
        dehazed = dehazed + residual  
        dehazed = torch.clamp(dehazed, 0, 1)
        
        dp = self.depth_dec4(bn, f4)
        dp = self.depth_dec3(dp, f3)
        dp = self.depth_dec2(dp, f2)
        dp = self.depth_dec1(dp, f1)
        depth = self.depth_out(dp)
        depth = torch.sigmoid(depth)
        
        return dehazed, depth
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)