import torch
from torch import nn
torch.manual_seed(1)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=2):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.block1 = Block(in_channels, in_channels // 4, kernel_size=1, padding=0, stride=1)
        self.block2 = Block(in_channels // 4, in_channels // 4, kernel_size=3, padding=1, stride=1)
        self.block3 = Block(in_channels // 4, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        skip = x
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = x + skip
        
        return x
    
    
class Model(nn.Module):
    def __init__(self, hid_dim=64):
        super().__init__()
        self.hid_dim = hid_dim
        self.emb_dim = 4 * hid_dim
        
        self.backbone = nn.Sequential(
            Block(3, hid_dim),                     # 25 * 100
            
            ResBlock(hid_dim, hid_dim),
            ResBlock(hid_dim, hid_dim),
            Block(hid_dim, 2 * hid_dim),           # 13 * 50
            
            ResBlock(2 * hid_dim, 2 * hid_dim),
            ResBlock(2 * hid_dim, 2 * hid_dim),
            Block(2 * hid_dim, 4 * hid_dim),       # 7 * 25
            
            ResBlock(4 * hid_dim, 4 * hid_dim),
            ResBlock(4 * hid_dim, 4 * hid_dim),
            Block(4 * hid_dim, 8 * hid_dim),       # 4 * 13
            
            ResBlock(8 * hid_dim, 8 * hid_dim),
            ResBlock(8 * hid_dim, 8 * hid_dim),
            Block(8 * hid_dim, 8 * hid_dim, kernel_size=4, padding=0),              # 1 * 5
            Block(8 * hid_dim, self.emb_dim, kernel_size=1, padding=0, stride=1)
        )

        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.emb_dim, 10 + 26),
            )
            for _ in range(5)
        ])

    def forward(self, x):
        z = self.backbone(x)

        logits_per_letter = []
        for i in range(5):
            logits = self.classifiers[i](z[:, :, 0, i])
            logits_per_letter.append(logits)
        
        return logits_per_letter
        