import torch.nn as nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import sys
sys.path.append("..")
from posenc import positionalencoding2d, positionalencoding2d2
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils import clip_grad_norm_
    
class DenseNet201(nn.Module):
    def __init__(self, num_classes, embed_dim, dropout = 0.1, grad_clip_value=1.0):
        self.embed_dim = embed_dim
        self.num_classes =  num_classes
        super(DenseNet201, self).__init__()

        self.features = models.densenet201(pretrained=True).features       #Feature extractor all except the final layer
        self.num_features = self.features[-1].num_features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(self.num_features, embed_dim)           
        self.gelu1 = nn.GELU()
        self.out_layer = nn.Linear(embed_dim, num_classes * embed_dim)
        self.gelu2 = nn.GELU()
        self.dropout = dropout
        self.grad_clip_value = grad_clip_value
        
    def clip_gradients(self):
        clip_grad_norm_(self.parameters(), self.grad_clip_value)

    def forward(self, x):                                              # [batch, channel, height, width]
        x = self.features(x)                                           # [batch, num_features, reduced_height, reduced_width]
        x = self.avgpool(x)                                            # [batch_size, num_features, 1, 1]
        x = x.view(x.size(0), -1)                                      # [batch_size, num_features]
        x = self.gelu1(self.proj(x))                                   # [batch_size, embed_dim]
        x = F.dropout(x, self.dropout, training=self.training)         # [batch_size, embed_dim]
        x = self.gelu2(self.out_layer(x)).view(-1, self.num_classes, self.embed_dim) # [batch_size, num_classes, embed_dim]

        return x
    
class CTranEncoder(nn.Module):
    def __init__(self, num_classes, in_channels, embed_dim, num_layers, num_heads, backbone, dropout = 0.1):
        super(CTranEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_classes =36
        self.in_channels = in_channels
        self.dropout = dropout
        self.backbone = backbone
        
        self.positional_encoding = nn.Parameter(torch.mean(positionalencoding2d(embed_dim, height=384, width=384), dim=0, 
        keepdim=True).unsqueeze(0).expand(1, in_channels, -1, -1), requires_grad=False)        # [1, channel, height, width]

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=2048, batch_first= True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers) 

        self.proj = nn.Linear(self.num_classes * embed_dim, 2*embed_dim)
        self.bn = nn.BatchNorm1d(2 * embed_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(2*embed_dim, num_classes), nn.ReLU())

        
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
    
    
class CTranEncodert(nn.Module):
    def __init__(self, num_classes, in_channels, embed_dim, num_layers, num_heads, backbone, dropout = 0.01):
        super(CTranEncodert, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_patches = 36
        self.in_channels = in_channels
        self.dropout = dropout
        self.backbone = backbone
        
        self.pos_embed = nn.Parameter(positionalencoding2d2(embed_dim, self.num_patches))        # [1, num_patches, embed_dim]

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=2048, batch_first= True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers) 

        self.proj1 = nn.Sequential(nn.GELU(), nn.Linear(embed_dim, int(embed_dim/2)))
        self.bn = nn.BatchNorm1d(int(embed_dim/2), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.proj2 = nn.Sequential(nn.GELU(), nn.Linear(int(embed_dim/2), 128))
        self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(self.num_patches, num_classes))

        
    def forward(self, x):                                                      # [batch, channel, height, width]
        x = self.backbone(x)                                                   # [batch, num_patches, embed_dim]
        # print(x.shape)
        x = x + self.pos_embed                                                 # [batch, num_patches, embed_dim]
        # print(x.shape)
        x = self.transformer(x)                                                # [batch, num_patches, embed_dim]
        x = self.proj1(x)                                                      # [batch, num_patches, 2*embed_dim]
        # print(x.shape)
        x = self.bn(x.transpose(1,2)).transpose(1,2)
        x = self.proj2(x)                                                      # [batch, num_patches, 128]
        # print(x.shape)
        x = torch.flatten(torch.mean(x, dim=-1, keepdim=True), start_dim = 1)  # [batch, num_patches]
        # print(x.shape)
        x = self.mlp(x)                                                        # [batch, num_classes]
        # print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)  
        return x  