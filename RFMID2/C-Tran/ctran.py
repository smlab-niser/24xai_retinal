import torch.nn as nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import sys
sys.path.append("..")
from posenc import positionalencoding2d
import torch.nn.functional as F
    

class CTranEncoder(nn.Module):
    def __init__(self, num_classes, in_channels, embed_dim, num_layers, num_heads, backbone, dropout = 0.1):
        super(CTranEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout = dropout
        self.backbone = backbone
        
        self.positional_encoding = nn.Parameter(torch.mean(positionalencoding2d(embed_dim, height=384, width=384), dim=0, 
        keepdim=True).unsqueeze(0).expand(1, in_channels, -1, -1), requires_grad=False)        # [1, channel, height, width]

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=2048)
        self.transformer = TransformerEncoder(encoder_layer, num_layers) 

        self.proj = nn.Linear(num_classes * embed_dim, 2*embed_dim)
        self.bn = nn.BatchNorm1d(2 * embed_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mlp = nn.Sequential(nn.Linear(2*embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, num_classes))
        #self.mlp = nn.Sequential(nn.GELU(), nn.Linear(2*embed_dim, num_classes))
        
    def forward(self, x):                                                    # [batch, channel, height, width]
        x = self.backbone(x)                                                # [batch, num_classes, embed_dim]
        with torch.no_grad(): posenc = self.backbone(self.positional_encoding) # [1, num_classes, embed_dim]
        # print(x[0, 0, :10], posenc[0, 0, :10])
        x = x + posenc                                                       # [batch, num_classes, embed_dim]
        x = self.transformer(x)                                              # [batch, num_classes, embed_dim]
        x = torch.flatten(x, start_dim=1)                                    # [batch, num_classes * embed_dim]
        x = self.proj(x)                                                     # [batch, 2 * embed_dim]
        x = self.bn(x)                                                       # [batch, 2 * embed_dim]
        x = self.mlp(x)                                                      # [batch, num_classes]
        x = F.dropout(x, self.dropout, training=self.training)  
        return x  