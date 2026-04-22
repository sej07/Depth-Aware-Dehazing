import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthAttention(nn.Module):
    
    def __init__(self, channels, attention_type='learned'):
        super().__init__()
        
        self.attention_type = attention_type
        
        if attention_type == 'learned':
            
            self.attention_net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        
    def forward(self, features, depth):
        B, C, H, W = features.shape

        if depth.shape[2:] != features.shape[2:]:
            depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)
        
        if self.attention_type == 'direct':
            attention = 1.0 - depth 
            attention = attention.expand(-1, C, -1, -1) 
            
        elif self.attention_type == 'learned':
            attention = self.attention_net(depth) 
        
        return features * attention + features


class DepthGuidedFFANet(nn.Module):
    
    def __init__(self, in_channels=3, channels=64, num_groups=3, num_blocks=6,
                 attention_type='learned', injection_points=['middle']):
        super().__init__()
        
        from .ffanet import FAGroup
        
        self.in_channels = in_channels
        self.injection_points = injection_points if injection_points != ['all'] else ['early', 'middle', 'late']
        

        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        
        self.groups = nn.ModuleList([
            FAGroup(channels, num_blocks) for _ in range(num_groups)
        ])

        if 'early' in self.injection_points:
            self.depth_attn_early = DepthAttention(channels, attention_type)
        if 'middle' in self.injection_points:
            self.depth_attn_middle = nn.ModuleList([
                DepthAttention(channels, attention_type) for _ in range(num_groups)
            ])
        if 'late' in self.injection_points:
            self.depth_attn_late = DepthAttention(channels, attention_type)
        
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
                    
    def forward(self, x, depth):
        residual = x
        
        f = self.conv_in(x)
        
        if 'early' in self.injection_points:
            f = self.depth_attn_early(f, depth)
        
        group_outputs = []
        for i, group in enumerate(self.groups):
            f = group(f)
            
            if 'middle' in self.injection_points:
                f = self.depth_attn_middle[i](f, depth)
                
            group_outputs.append(f)
        
        fused = torch.cat(group_outputs, dim=1)
        fused = self.conv_fusion(fused)
        
        if 'late' in self.injection_points:
            fused = self.depth_attn_late(fused, depth)
        
        out = self.conv_out(fused)
        out = out + residual
        out = torch.clamp(out, 0, 1)
        
        return out
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)