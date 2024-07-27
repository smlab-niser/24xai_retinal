import torch.nn as nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import sys
import torch.nn.functional as F
sys.path.append("..")
from posenc import positionalencoding2d



# Variant 1
class CTranEncoder(nn.Module):
    def __init__(self, num_classes, in_channels, embed_dim, num_layers, num_heads, backbone, dropout = 0.1):
        super(CTranEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout = dropout
        self.backbone = backbone
        
        self.positional_encoding = nn.Parameter(positionalencoding2d(embed_dim, height=384, width=384),
        requires_grad=False).permute(1, 2, 0).mean(dim=(0, 1), keepdim=True).expand(1, num_classes, embed_dim)
        print(self.positional_encoding.shape)

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=2048)
        self.transformer = TransformerEncoder(encoder_layer, num_layers) 

        self.proj = nn.Linear(num_classes * embed_dim, 2*embed_dim)
        self.bn = nn.BatchNorm1d(2 * embed_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mlp = nn.Sequential(nn.Linear(2*embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, num_classes))

        
    def forward(self, x):                                                    # [batch, channel, height, width]
        x = self.backbone(x)                                                # [batch, num_classes, embed_dim]
        # with torch.no_grad: posenc = self.backbone(self.positional_encoding) # [1, num_classes, embed_dim]
        posenc = self.positional_encoding.to(x.device)
        x = x + posenc                                                       # [batch, num_classes, embed_dim]
        x = self.transformer(x)                                              # [batch, num_classes, embed_dim]
        x = torch.flatten(x, start_dim=1)                                   # [batch, num_classes * embed_dim]
        x = self.proj(x)                                                     # [batch, 2 * embed_dim]
        x = self.bn(x)                                                       # [batch, 2 * embed_dim]
        x = self.mlp(x)                                                      # [batch, num_classes]
        x = F.dropout(x, self.dropout, training=self.training)  
        return x  
    

#Variant 2
class CTranEncoder2(nn.Module):
    def __init__(self, num_classes, embed_dim, num_layers, num_heads, backbone1, backbone2):
        super(CTranEncoder2, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.backbone1 = backbone1
        self.backbone2 = backbone2

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=2048)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
       
        # Initialize the final classification layer
        self.fc = nn.Linear(embed_dim, num_classes)
        
        # Initialize the projection layer
        self.proj = nn.Linear(2*embed_dim, embed_dim)
        
        # Initialize the batch normalization layer
        self.bn = nn.BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.positional_encoding = nn.Parameter(positionalencoding2d(embed_dim, height=384, width=384),
        requires_grad=False).permute(1, 2, 0).mean(dim=(0, 1), keepdim=True).expand(1, num_classes, embed_dim)
        
        # Initialize the linear layer for dimensionality reduction
        self.proj = nn.Linear(2* num_classes * embed_dim, 2*embed_dim)
        self.bn = nn.BatchNorm1d(2 * embed_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mlp = nn.Sequential(nn.Linear(2*embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, num_classes))

    def forward(self, x): 
        y = self.backbone2(x)
        x = self.backbone1(x)

        x = torch.cat((x, y), dim=1)
        # print("x size after concat:", x.size())
        
        posenc = self.positional_encoding.to(x.device) # [1, num_classes, embed_dim]
        x = x + torch.cat((posenc, posenc), dim=1)
        #print("x size before transformer:", x.size())
        
        x = self.transformer(x)
        x = torch.flatten(x, start_dim=1)                                   # [batch, num_classes * embed_dim]
        x = self.proj(x)                                                     # [batch, 2 * embed_dim]
        x = self.bn(x)                                                       # [batch, 2 * embed_dim]
        x = self.mlp(x)                                                      # [batch, num_classes]
        return x