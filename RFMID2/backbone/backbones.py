import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import timm

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
        # self.grad_clip_value = grad_clip_value
        
    def clip_gradients(self):
        clip_grad_norm_(self.parameters(), self.grad_clip_value)

    def forward(self, x):                                              # [batch, channel, height, width]
        x = self.features(x)                                           # [batch, num_features, reduced_height, reduced_width]
        x = self.avgpool(x)                                            # [batch_size, num_features, 1, 1]
        x = x.view(x.size(0), -1)                                      # [batch_size, num_features]
        x = self.gelu1(self.proj(x))                                   # [batch_size, embed_dim]
        # x = F.dropout(x, self.dropout, training=self.training)         # [batch_size, embed_dim]
        x = self.gelu2(self.out_layer(x)).view(-1, self.num_classes, self.embed_dim) # [batch_size, num_classes, embed_dim]

        return x
    

class ResNet152d(nn.Module):
    def __init__(self, num_classes, embed_dim, dropout = 0.1, grad_clip_value=1.0):
        self.embed_dim = embed_dim
        self.num_classes =  num_classes
        super(ResNet152d, self).__init__()

        self.features = timm.create_model('resnet152d', pretrained=True)
        self.num_features = self.features.fc.in_features
        self.features.fc = nn.Identity() # remove the original fully connected layer
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(self.num_features, embed_dim)           
        self.gelu1 = nn.GELU()
        self.out_layer = nn.Linear(embed_dim, num_classes * embed_dim)
        self.gelu2 = nn.GELU()
        self.dropout = dropout
        self.grad_clip_value = grad_clip_value
        
    def clip_gradients(self):
        clip_grad_norm_(self.parameters(), self.grad_clip_value)

    def forward(self, x):                                              # [batch, channel, height, width]
        x = self.features(x)                                           # [batch, num_features]
        x = self.gelu1(self.proj(x))                                   # [batch_size, embed_dim]
        x = F.dropout(x, self.dropout, training=self.training)         # [batch_size, embed_dim]
        x = self.gelu2(self.out_layer(x)).view(-1, self.num_classes, self.embed_dim) # [batch_size, num_classes, embed_dim]

        return x
    
    
    
class DenseNet201b(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(DenseNet201b, self).__init__()

        self.features = models.densenet201(pretrained=True).features 
        self.num_features = self.features[-1].num_features 
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))   
        self.proj = nn.Linear(self.num_features, embed_dim)
        self.gelu = nn.GELU()
        self.dropout = dropout

    def forward(self, x):                                               # [batch, channel, height, width]
        x = self.features(x)                                            # [batch, num_features, reduced_height, reduced_width]
        x = self.maxpool(x)                                             # [batch, num_features, 1, 1]
        x = x.view(x.size(0), -1)                                       # [batch, num_features]
        x = self.gelu(self.proj(x))                                     # [batch, embed_dim]
        x = F.dropout(x, self.dropout, training=self.training)
        
        return x
    


class DenseNet201t(nn.Module):
    def __init__(self, num_classes, embed_dim, patch_size = 2, dropout = 0.05, grad_clip_value=1.0, feature_layer=7):
        self.embed_dim = embed_dim
        self.num_classes =  num_classes
        self.ps = patch_size
        super(DenseNet201t, self).__init__()
        self.features = models.densenet201(pretrained=True).features       #Feature extractor all except the final layer
        self.num_features = self.features[-1].num_features
        print(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(self.num_features, 2*embed_dim)           
        self.relu1 = nn.ReLU()
        self.out_layer = nn.Linear(embed_dim*2, embed_dim)
        self.relu2 = nn.ReLU()
        self.dropout = dropout
        # self.grad_clip_value = grad_clip_value
        
    def clip_gradients(self):
        clip_grad_norm_(self.parameters(), self.grad_clip_value)

    def forward(self, x):                                              # [batch, channel, height, width]
        x = self.features(x)                                           # [batch, num_features, reduced_height, reduced_width]
        # print(x.shape)
        ps = self.ps
        x = x.unfold(2, ps, ps).unfold(3, ps, ps).contiguous().reshape(x.size(0), x.size(1), -1, ps, ps).permute(0,2,1,3,4)
        # print(x.shape)
        x = self.avgpool(x)                                            # [batch_size, N, num_features, 1, 1]
        # print(x.shape)
        x = x.view(x.size(0), x.size(1), -1)                                      # [batch_size, N, num_features]
        # print(x.shape)
        x = self.relu1(self.proj(x))                                   # [batch_size, N, embed_dim]
        # print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)         # [batch_size, embed_dim]
        #x = self.gelu2(self.out_layer(x)).view(-1, self.num_classes, self.embed_dim) # [batch_size, num_classes, embed_dim]
        x = self.relu2(self.out_layer(x))
        return x