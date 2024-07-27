
import torch
import torch.nn as nn

class CLSToken(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)
        x = x.unsqueeze(1)

        return x
    
class CLSToken2(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()

        self.conv= nn.Conv2d(in_channels, embed_dim, kernel_size = patch_size, stride = patch_size)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.conv(x)                     # [batch, embed_dim, H, W]
        x = torch.mean(x, dim=(2, 3))        # [batch, embed_dim]
        x = self.fc(x)                       # [batch, embed_dim]
        x = x.unsqueeze(1)                   # [batch, 1, embed_dim]

        return x