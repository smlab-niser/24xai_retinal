import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import sys
sys.path.extend(["..", "../backbone","../.."])
import torch.nn.functional as F
from posenc import posencDPD
from clstoken import CLSToken2


class IEViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, 
                 num_layers, dim_feedforward, mlp_dim, backbone, dropout, device):
        super().__init__()

        assert img_size % patch_size == 0, 'image size must be divisible by patch size'
       
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_patches = int((img_size / patch_size) ** 2)
        self.old_subpatch_counts = [1]*self.num_patches

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.cls_token = CLSToken2(in_channels, embed_dim, patch_size) 
        self.pos_embed = None
        
        min_mean = (self.num_patches - 1) / 3
        max_mean = 2 * (self.num_patches - 1) / 3
        self.sigma = nn.Parameter(torch.tensor(2.0)).to(device)
        self.mean = nn.Parameter(torch.tensor((self.num_patches - 1) / 2, dtype=torch.float32)).to(device)
        self.mean.data = torch.clamp(self.mean.data, min_mean, max_mean) 
        
        self.LIW = LearnableImportanceWeights(self.num_patches, device)
        self.importance_weights = nn.Parameter(F.softmax(torch.ones(1, self.num_patches), dim=1).squeeze(0))
        self.cls_posenc = torch.zeros((1, 1, self.embed_dim)).to(device)
        
        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=dim_feedforward)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.backbone = backbone
        
        self.dynamic_lin = DynamicLinear((self.num_patches + 1 + num_layers) * self.embed_dim, mlp_dim)
        self.mlp_head = nn.Sequential(nn.GELU(), nn.Linear(mlp_dim, num_classes))
        self.dropout = dropout
        
    def generate_positional_encoding(self, subpatch_counts, device):
        embed_dim = self.embed_dim   
        pos_encoding = posencDPD(embed_dim, subpatch_counts)
        self.pos_embed = pos_encoding.to(device)


    def forward(self, x):
        
        emd = self.embed_dim
        sigma, mean = self.sigma, self.mean
        ximg = self.backbone(x).unsqueeze(1) # [batch, 1, embed_dim]
        
        importance_weights, kld = self.LIW(self.importance_weights, sigma, mean)
        # print(self.importance_weights, importance_weights == self.importance_weights)
        patch_em, subpatch_counts = self.patch_embed(x, importance_weights, self.num_patches) # [batch, N, embed_dim]
        # print(patch_em.shape)
        
        self.generate_positional_encoding(subpatch_counts, ximg.device)                            # [1, N, embed_dim]
        # print(subdivided_patch_indices)
        
        cls_token = self.cls_token(x)                                                              # [batch, 1, embed_dim]
        x = torch.cat((cls_token, patch_em), dim=1)                                                # [batch, N+1, embed_dim]

        self.pos_embed = torch.cat((self.cls_posenc, self.pos_embed), dim=1)             # [1, N+1, embed_dim]  (Extend pos_embed dimensions)
        x = x + self.pos_embed                                                                     # [batch, N+1, embed_dim]

        for i in range(self.num_layers):
            transformer_layer = self.transformer.layers[i]
            x = transformer_layer(x)                               
            x = torch.cat((ximg, x), dim=1)         
        
        x = self.layer_norm(x)          # [batch, N+1+L, embed_dim]                                             
        x = x.flatten(1)                # [batch, (N+1+L) * embed_dim]   
        
        weight = self.dynamic_lin.weight.clone()
        ini = self.num_layers
        

        if self.old_subpatch_counts != subpatch_counts:

            ini = (self.num_layers + 1) * emd
            final_weight = [weight[:,: ini]]

            for i, num in enumerate(subpatch_counts):
                old_num = self.old_subpatch_counts[i]
                weight_slice = weight[:,ini: (ini + old_num* emd)]
                # print("weight_slice: ", weight_slice.shape)

                if num != old_num:  
                    scale = num / old_num
                    new_weight = torch.empty(0).to(x.device)
                    
                    if num > old_num:
                        # print(i, "poop")
                        # print(weight_slice.shape)
                        for k in range(old_num):
                            section = weight_slice[:, k * emd:(k + 1) * emd]
                            new_w = torch.cat([section] * int(scale), dim=-1)
                            new_weight = torch.cat((new_weight, new_w), dim=-1)
                            # print("section, new_weight: ", section.shape, new_weight.shape)
                    else:
                        scale = int(old_num / num)
                        # print(i, "piss")
                        for k in range(num):
                            sliced_parts = torch.chunk(weight_slice[:, k * emd * scale: (k + 1) * emd * scale], chunks=scale, dim=-1)
                            section = torch.mean(torch.stack(sliced_parts, dim=-1), dim=-1)
                            new_weight = torch.cat((new_weight, section), dim=-1)
                            # print("section, new_weight: ", torch.stack(sliced_parts, dim=-1).shape, new_weight.shape)
                
                else: new_weight = weight_slice
                
                final_weight.append(new_weight)
                ini += old_num * emd
            # print(len(final_weight))
            final_weight = torch.cat(final_weight, dim=-1)
        
        else: final_weight = weight
        # print(final_weight.shape)
        self.old_subpatch_counts = subpatch_counts    
        
        self.dynamic_lin.set_weight(final_weight)
        x = self.dynamic_lin(x)
        x = self.mlp_head(x)
        x = F.dropout(x, self.dropout, training=self.training)

        return x, kld



class DynamicLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(DynamicLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)

    def set_weight(self, weight):
        self.weight = nn.Parameter(weight)
        

class LearnableImportanceWeights(nn.Module):
    def __init__(self, num_patches, device):
        super(LearnableImportanceWeights, self).__init__()
        self.num_patches = num_patches
        self.register_buffer("patch_positions", torch.arange(num_patches).float().to(device))
        self.fc = nn.Linear(self.num_patches, 128)
        self.output = nn.Linear(128, self.num_patches)
        self.softmax = nn.Softmax(dim=0)
        self.device = device

    def forward(self, x, sigma, mean):
        # print(self.sigma.item(), self.mean.item())
        eps = 1e-41
        gaussian_weights = nn.Parameter(torch.exp(-0.5 * ((self.patch_positions - mean) / sigma) ** 2)+eps).to(x.device)
        gw = gaussian_weights / gaussian_weights.sum()
        x = F.relu(self.fc(x))
        x = self.output(x)
        new_weights = self.softmax(x)
        log_new_weights = new_weights.log()
        kld = nn.functional.kl_div(log_new_weights, gw, reduction='sum')

        return new_weights, kld
    
def sigmoid_round2(x, temp=1.0):
    return 32 * torch.sigmoid((x - 16) / temp) + 16 * torch.sigmoid(-(x - 24) / temp) + 8 * torch.sigmoid(-(x - 28) / temp)

def sigmoid_round(x, temp=1.0):
    x = torch.sigmoid((x - 24) / temp)
    return torch.round(32 * x + 16 * (1 - x)).long() 


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    

    def forward(self, x, importance_w, num_patches):
        
        importance_weights = importance_w.clone()
        # Calculate new patch sizes based on importance weights
        average_weight = importance_weights.mean(dim=0)
        # print("average_weight = ", average_weight)
        scaling_factor = 12
        ps = self.patch_size

        # scaled_weights = importance_weights.pow(scaling_factor)
        # new_patch_sizes = (ps * average_weight.pow(scaling_factor) / scaled_weights).clamp(max=ps)
        scaled_weights = importance_weights.pow(scaling_factor)
        new_patch_sizes = (ps * average_weight.pow(scaling_factor) / scaled_weights)
        # print(new_patch_sizes)
        # new_patch_sizes = [min([32, 16, 8], key=lambda x: abs(x - num)) for num in new_patch_sizes]
        new_patch_sizes = sigmoid_round(new_patch_sizes)
        print(new_patch_sizes)
        
        new_patches= None    
        subdivided_patch_indices = []
        subpatch_counts = []
        H = x.shape[3]
        # print(new_patch_sizes.size())

        for i in range(num_patches):
            patch_size = new_patch_sizes[i]
            
            if (patch_size < ps) and (patch_size > 7):
                
                subdivided_patch_indices.append(i)
                num_subpatches = int((ps / patch_size) ** 2) # In row or column
                row_subpatches = int((ps / patch_size))
                subpatch_size = patch_size
                patches = None
                
                for j in range(num_subpatches):
                    
                    ini_col = (i * ps) % H
                    ini_row = ((i * ps) // H) * ps
                    start_col = ini_col + ((j * subpatch_size) % (row_subpatches * subpatch_size))
                    start_row = ini_row + ((j * subpatch_size) // (row_subpatches * subpatch_size)) * subpatch_size
                    end_row = start_row + subpatch_size
                    end_col = start_col + subpatch_size
                    print(i, j, start_row, end_row, start_col, end_col, type(end_col.item()))
                    patch = x[:, :, start_row:end_row, start_col:end_col]
                    
                    
                    kernel_divisor = ps // patch_size # Calculate the number of times the kernel should be divided in each dimension  
                    original_weights = self.patch_embed.weight # Create a new view of the original kernel 
                    modified_weights = original_weights.view(self.embed_dim, self.in_channels, 
                            kernel_divisor, kernel_divisor, patch_size, patch_size) # Reshape the kernel to match the new patch size [embed_dim, in_channels, ps, ps] ---> [embed_dim, in_channels, kernel_div, kernel_div, patch_size, patch_size]
                    
                    modified_weights = modified_weights.mean(dim=2).mean(dim=2)  # Mean over the dimensions where the kernel was divided [embed_dim, in_channels, patch_size, patch_size]
                    conv_layer = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=patch_size, 
                                           stride=patch_size)  # Create a new convolutional layer with the modified kernel size [embed_dim, embed_dim, patch_size, patch_size]
                    conv_layer.bias = self.patch_embed.bias
                    conv_layer.weight = nn.Parameter(modified_weights)  # Assign the modified weights to the new conv layer 
                    embed = conv_layer(patch)                           # [batch, embed_dim, patch_size, patch_size]
                    embed = self.avgpool(embed)                         # [batch, embed_dim, 1, 1]
                    embed = embed.flatten(2).transpose(1, 2)            # [batch, 1, embed_dim]
                    
                    if patches is None: patches = embed
                    else:  patches = torch.cat((patches, embed),1)
                
                if new_patches is None: new_patches = patches
                else:  new_patches = torch.cat((new_patches, patches),1)  

                subpatch_counts.append(num_subpatches)
            else:
                subpatch_counts.append(1)
        
        # print(subpatch_counts)
        final_patches = torch.empty(0).to(x.device)
        patch_index = 0

        for i in range(num_patches):
            if i in subdivided_patch_indices:
                num_subpatches = subpatch_counts[i]
                final_patches = torch.cat((final_patches, new_patches[:,patch_index:patch_index + num_subpatches,:]), 1)
                patch_index += num_subpatches
            else:
                ini_col = (i * ps) % H
                ini_row = ((i * ps) // H) * ps
                end_row = ini_row + ps
                end_col = ini_col + ps
                # print(i, ini_row, end_row, ini_col, end_col)
                embed = self.patch_embed(x[:, :, ini_row:end_row, ini_col:end_col])
                embed = self.avgpool(embed)    
                embed = embed.flatten(2).transpose(1, 2)
                final_patches = torch.cat((final_patches, embed), 1)

        return final_patches, subpatch_counts


