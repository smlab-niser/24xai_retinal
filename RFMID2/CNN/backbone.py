import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
# from efficientnet_pytorch import EfficientNet
import timm

class DenseNet121(nn.Module):
    def __init__(self, num_classes, embed_dim, dropout=0.1):
        super(DenseNet121, self).__init__()
        
        self.features = models.densenet121(pretrained=True).features
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DenseNet201(nn.Module):
    def __init__(self, num_classes, embed_dim, dropout=0.1):
        super(DenseNet201, self).__init__()
        
        self.features = models.densenet201(pretrained=True).features
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1920, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class ResNet152V2(nn.Module):
    def __init__(self, num_classes):
        super(ResNet152V2, self).__init__()

        self.model = timm.create_model("resnet152d", pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class DenseNet201b(nn.Module):
    def __init__(self, num_classes, embed_dim, dropout=0.1):
        super(DenseNet201b, self).__init__()

        self.features = models.densenet201(pretrained=True).features 
        self.num_features = self.features[-1].num_features 
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))   
        self.proj = nn.Linear(self.num_features, embed_dim)
        self.gelu = nn.GELU()
        self.out_layer = nn.Linear(embed_dim, num_classes)
        self.gelu2 = nn.GELU()
        self.dropout = dropout

    def forward(self, x):                                               # [batch, channel, height, width]
        x = self.features(x)                                            # [batch, num_features, reduced_height, reduced_width]
        x = self.maxpool(x)                                             # [batch, num_features, 1, 1]
        x = x.view(x.size(0), -1)                                       # [batch, num_features]
        x = self.gelu(self.proj(x))                                     # [batch, embed_dim]
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gelu2(self.out_layer(x))
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
        self.out_layer = nn.Linear(embed_dim, num_classes)
        self.gelu2 = nn.GELU()
        self.dropout = dropout
        self.grad_clip_value = grad_clip_value
        
    def clip_gradients(self):
        clip_grad_norm_(self.parameters(), self.grad_clip_value)

    def forward(self, x):                                              # [batch, channel, height, width]
        x = self.features(x)                                           # [batch, num_features]
        x = self.gelu1(self.proj(x))                                   # [batch_size, embed_dim]
        x = F.dropout(x, self.dropout, training=self.training)         # [batch_size, embed_dim]
        x = self.gelu2(self.out_layer(x))

        return x
    
class DenseNet121_2(nn.Module):
    def __init__(self, num_classes, embed_dim, dropout=0.1):
        super(DenseNet121_2, self).__init__()
        
        self.features = models.densenet121(pretrained=True).features
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        return x
    
class DenseNet201_2(nn.Module):
    def __init__(self, num_classes, embed_dim, dropout=0.1):
        super(DenseNet201_2, self).__init__()
        
        self.features = models.densenet201(pretrained=True).features
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1920, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        return x
    
class ResNet152d_2(nn.Module):
    def __init__(self,  num_classes, embed_dim, dropout=0.1):
        super(ResNet152d_2, self).__init__()

        # self.features = timm.create_model("resnet152d", pretrained=True)
        self.features = timm.create_model('resnet152d', pretrained=True, num_classes = num_classes)
        # self.num_features = self.features.fc.in_features
        # self.features.fc = nn.Identity()
        self.fc3 = nn.Linear(num_classes, num_classes)
        self.relu = nn.ReLU()
        # self.fc = nn.Linear(self.num_features, num_classes)
        # self.fc = nn.Linear(self.num_features, embed_dim)
        self.relu2 = nn.ReLU()
        # self.fc2 = nn.Linear(embed_dim, num_classes)
        # self.dropout = dropout

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x =self.fc(x)
        x =self.relu2(x)
        x =self.fc3(x)
        x = self.relu(x)
        # print(x.shape)
        return x

# class ResNet152d_2(nn.Module):
#     def __init__(self, num_classes, embed_dim, dropout=0.1):
#         super(ResNet152d_2, self).__init__()
#         self.features = timm.create_model('resnet152d', pretrained=True, num_classes=num_classes)
#         self.num_features = self.features.get_classifier().in_features
#         self.features.reset_classifier(0)
#         self.fc = nn.Linear(self.num_features, embed_dim)
#         self.bn = nn.BatchNorm1d(embed_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(embed_dim, num_classes)
#         self.relu2 = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.fc(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.relu2(x)
#         return x


class EfficientNetV2Small(nn.Module):
    def __init__(self, num_classes, embed_dim, dropout=0.1):
        super(EfficientNetV2Small, self).__init__()
        
        self.model = timm.create_model("tf_efficientnetv2_s", pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class EfficientNetV2Small_2(nn.Module):
    def __init__(self, num_classes, embed_dim, dropout=0.1):
        super(EfficientNetV2Small_2, self).__init__()
        
        self.features = timm.create_model("tf_efficientnetv2_s", pretrained=True, num_classes=num_classes)
        self.fc = nn.Linear(num_classes, num_classes)
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.features(x)
        x =self.relu(x)
        x =self.fc(x)
        x = self.relu2(x)
        return x

# Dictionary containing the models
backbone = {
    'densenet121': DenseNet121,
    'densenet121-2': DenseNet121_2,
    'densenet201': DenseNet201,
    'densenet201-2': DenseNet201_2,
    'resnet152v2': ResNet152V2,
    'resnet152d': ResNet152d,
    'resnet152d-2': ResNet152d_2,
    'densenet201b': DenseNet201b,
    'effnet': EfficientNetV2Small,
    'effnet-2': EfficientNetV2Small_2
}