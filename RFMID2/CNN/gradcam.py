import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import sys
sys.path.extend(["..", "../backbone"])
import torchvision.transforms as transforms
from transform import Transform
from data import RetinaDataset
from heatmap_gen import generate_hm
torch.cuda.empty_cache()

device = torch.device("cuda:0")
model = torch.load('models/dn1.pth')
model.to(device)

df = pd.read_csv('../../../data/GT-main/set1/all.csv')
image_names = ["im0151.png", "js888.jpg", "1044.png","283.png","aria_d_25_33.tif","rf653.jpg", "2.png","18.png","1370.png","js61.jpg","rb98.jpg","rb27.jpg","im0007.png","js245.jpg","1550.png"]
save_loc = "./gradcam"
label_set = [5,5,8,8,0,0,0,0,3,3,20,20,15,16,16]
idx = df[df['ID'].isin([im.split('.')[0] for im in image_names])].index.tolist()
print(idx)

data_dir = '../../../data/GT-main'
batch_size = 16
image_size = 384
num_labels = 21
num_workers = 4
thresholds = [0.5] * num_labels
phase = 'all'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
threshold = [0.65, 0.7, 0.49, 0.43, 0.66, 0.78, 0.26, 0.21, 0.38, 0.82, 0.64, 0.9, 0.38, 0.59, 0.86, 0.31, 0.51, 0.69, 0.91, 0.23, 0.37]
class_names = ["DR","NORMAL","MH","ODC","TSLN","ARMD","DN","MYA","BRVO","ODP","CRVO","CNV","RS","ODE","LS","CSR","HTR","ASR","CRS","OTHER","RB"]

transform = Transform(size=image_size, phase=phase)
label_list = []
image_list = []
dataset = RetinaDataset(data_dir=data_dir, split=phase, transform=transform)
for i in idx:
    image = dataset[i][0]
    label = dataset[i][1]
    image_list.append(image.permute(1,2,0).unsqueeze(0))
    label_list.append(label.unsqueeze(0))
    plt.figure(figsize=(1, 1))
    plt.imshow(image.permute(1,2,0))
    plt.axis('off')
    plt.show()
    print(label)

images = torch.cat(image_list, dim=0)
labels = torch.cat(label_list, dim=0)
true_labels = labels.detach().cpu().numpy().astype(np.int32)
print(true_labels)
print(labels.shape, images.shape)

def predict(x):
    tmp = torch.tensor(x).to(device)
    tmp = tmp.permute(0,3,1,2)
    return torch.sigmoid(model(tmp))

# Define the inverse transformation function
inv_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
    transforms.Normalize(mean=(-1 * np.array(mean) / np.array(std)).tolist(),std=(1 / np.array(std)).tolist(),),
    transforms.Lambda(lambda x: x.permute(0, 2, 3, 1)),
])

# Apply the inverse transformation to the transformed image tensor
original_images = inv_transform(images)
original_images = original_images.numpy()

# Plot the original images
plt.figure(figsize=(1, 1))
for i in range(original_images.shape[0]):
    plt.subplot(1, original_images.shape[0], i + 1)
    plt.imshow(original_images[i])
    plt.axis('off')
    plt.title(f'Image {i + 1}')
plt.show()

target_layer = model.backbone.features.denseblock4.denselayer32.conv2
gradients = None
activations = []

def get_activations(module, input, output):
    global activations
    activations.append(output)

def activations_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output  # or grad_output, depending on what you need

target_layer.register_backward_hook(activations_hook)
target_layer.register_forward_hook(get_activations)

model.eval()
output = predict(images)

thresholds = [0.5]*num_labels
outputs = output.detach().cpu().numpy()
outputs_thresholded = (outputs > thresholds).astype(np.int32)
print(outputs_thresholded)

heatmaps = []

for i in range(len(image_names)):
    pred = output[i,label_set[i]]
    pred.backward(retain_graph=True)
    grads = gradients[0][i:i+1]
    print(activations[0][i:i+1].shape)
    activs = activations[0][i:i+1].detach()
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    print(activs.shape, activs.type, grads.shape, pooled_grads.shape, pooled_grads.type)
    
    for j in range(pooled_grads.size(0)):
        activs[:, j, :, :] *= pooled_grads[j]
    
    heatmap = torch.mean(activs, dim=1).squeeze().detach().cpu()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    plt.matshow(heatmap.squeeze())
    heatmaps.append(heatmap.numpy())
    print(heatmap.shape)

generate_hm(image_names, heatmaps, save_loc)
