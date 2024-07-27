import torch.nn as nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import sys
sys.path.append("..")
from posenc import positionalencoding2d

    
class IECTe(nn.Module):
    def __init__(self, num_classes, in_channels, embed_dim, num_layers, num_heads, mlp_dim, backbone1, backbone2):
        super(IECTe, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Initialize the backbone network
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.num_layers = num_layers 
        
        # Initialize the TransformerEncoder
        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=2048)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        # Initialize the positional encoding matrix
        self.positional_encoding = nn.Parameter(torch.mean(positionalencoding2d(embed_dim, height=384, width=384), dim=0, 
        keepdim=True).unsqueeze(0).expand(1, in_channels, -1, -1), requires_grad=False)        # [1, channel, height, width]
        
        # Initialize the layer normalization layer
        self.layer_norm = nn.LayerNorm(2*embed_dim)
        self.bn = nn.BatchNorm1d(2 * embed_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.proj = nn.Linear(2*(num_classes+num_layers) * embed_dim, 2*embed_dim)
      
        self.mlp_head = nn.Sequential(
            nn.Linear(2*embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes))
        
    def forward(self, x): 
        y = self.backbone2(x)
        x = self.backbone1(x)
        
        # Concatenate y to x along the num_classes dimension
        z = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat((x, y), dim=1)
        y = torch.mean(y, dim=1, keepdim=True)

        with torch.no_grad(): posenc = self.backbone1(self.positional_encoding) # [1, num_classes, embed_dim]
        x = x + torch.cat((posenc, posenc), dim=1)
        
        for i in range(self.num_layers):
            transformer_layer = self.transformer.layers[i]
            x = transformer_layer(x)                               
            x = torch.cat((x, y), dim=1)
            x = torch.cat((x, z), dim=1)  
        
        x = torch.flatten(x, start_dim=1)
        x = self.proj(x) 
        x = self.layer_norm(x)
        x = self.bn(x)
        x = self.mlp_head(x) 
        return x    



class IEeCT(nn.Module):
    def __init__(self, num_classes, in_channels, embed_dim, num_layers, num_heads, mlp_dim, backbone1, backbone2):
        super(IEeCT, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Initialize the backbone network
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.num_layers = num_layers
        
        # Initialize the TransformerEncoder
        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=2048)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        # Initialize the positional encoding matrix
        self.positional_encoding = nn.Parameter(torch.mean(positionalencoding2d(embed_dim, height=384, width=384), dim=0, 
        keepdim=True).unsqueeze(0).expand(1, in_channels, -1, -1), requires_grad=False)        # [1, channel, height, width]

        # Initialize the layer normalization layer
        self.layer_norm = nn.LayerNorm(2*embed_dim)
        self.bn = nn.BatchNorm1d(2 * embed_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.proj = nn.Linear((num_classes+num_layers) * embed_dim, 2*embed_dim)
      
        self.mlp_head = nn.Sequential(
            nn.Linear(2*embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes))

        
    def forward(self, x): 
        y = torch.mean(self.backbone2(x), dim=1, keepdim=True)
        x = self.backbone1(x)
    
        with torch.no_grad(): posenc = self.backbone1(self.positional_encoding) # [1, num_classes, embed_dim]
        x = x + posenc
        
        for i in range(self.num_layers):
            transformer_layer = self.transformer.layers[i]
            x = transformer_layer(x)                               
            x = torch.cat((x, y), dim=1)  
            
        x = torch.flatten(x, start_dim=1)
        x = self.proj(x) 
        x = self.layer_norm(x)
        x = self.bn(x)
        x = self.mlp_head(x) 
        return x   
    
    
    
    

# class CTranEncoder(nn.Module):
#     def __init__(self, num_classes, embed_dim, num_layers, num_heads, backbone1, backbone2):
#         super(CTranEncoder, self).__init__()
        
#         # Initialize the backbone network
#         self.backbone1 = backbone1
#         self.backbone2 = backbone2
        
#         # Initialize the TransformerEncoder
#         encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=2048)
#         self.transformer = TransformerEncoder(encoder_layer, num_layers)
       
#         # Initialize the final classification layer
#         self.fc = nn.Linear(embed_dim, num_classes)
        
#         # Initialize the projection layer
#         self.proj = nn.Linear(2*embed_dim, embed_dim)
        
#         # Initialize the batch normalization layer
#         self.bn = nn.BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
#         # Initialize the positional encoding matrix
#         self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, embed_dim))
#         nn.init.normal_(self.positional_encoding, std=0.02)
        
#         # Initialize the class token
#         self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         nn.init.normal_(self.class_token, std=0.02)
        
#     def forward(self, x): 
#         # Pass the input through the backbone network
#         # print("Before backbone:", x.shape) torch.Size([batch, channel, height, width])
#         y = self.backbone2(x)
#         x = self.backbone1(x)
#         # print("After backbone:", x.shape) torch.Size([num_classes, batch, embed_dim])
#         # Add the positional encoding to the input
#         x = x + self.positional_encoding[:, :x.size(1), :]
#         # print("After encoding:", x.shape)  torch.Size([num_classes, batch, embed_dim])
#         # Pass the output through the TransformerEncoder
#         x = self.transformer(x)
#         # print("After transform:", x.shape) torch.Size([num_classes, batch, embed_dim])
#         # Concatenate the first and last hidden states along the last dimension
#         x = torch.cat((x[0], x[-1]), dim=-1)
#         # print("After cat:", x.shape) torch.Size([batch, 2*embed_dim])
#         # Pass the output through the projection layer
#         x = self.proj(x) 
#         # print("After proj:", x.shape) torch.Size([batch, embed_dim])
#         # Add a seq_len dimension to x
#         x = x.unsqueeze(2)
#         # print("After unsqueeze:", x.shape) torch.Size([batch, embed_dim, 1]) 
#         # Pass the output through the batch normalization layer
#         x = self.bn(x.transpose(1, 2)).transpose(1, 2)
#         # print("After bn:", x.shape) torch.Size([batch, embed_dim, 1])
#         # Remove the seq_len dimension from x
#         x = x.squeeze(2)
#         # print("After squeeze:", x.shape) torch.Size([batch, embed_dim])
#         # Pass the output through the final classification layer
#         x = self.fc(x)
#         # print("After fc:", x.shape) torch.Size([batch, num_classes])
#         return x