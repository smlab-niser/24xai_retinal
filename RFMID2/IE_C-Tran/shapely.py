import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import sys
sys.path.extend(["..", "../backbone","../.."])
import torchvision.transforms as transforms
from transform import Transform
from data import RetinaDataset
import shap
torch.cuda.empty_cache()

device = torch.device("cpu")
#device = torch.device("cuda:0")
model = torch.load('models/iecte1.pth', map_location=torch.device('cpu'))
model.to(device)


df = pd.read_csv('./../../../data/GT-main/set1/all.csv')
image_names = ["aria_a_13_2","im0151","js888","1592"]
idx = df[df['ID'].isin(image_names)].index.tolist()
print(idx)

filename1 = './shap/ARMD/all-iecte1.png'
filename2 = './shap/ARMD/all-iecte1-info.txt'

data_dir = '../../../data/GT-main'
batch_size = 16
image_size = 384
num_labels = 21
num_workers = 4
thresholds = [0.5]*num_labels
phase= 'all'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
threshold = [0.5,  0.7,  0.83, 0.75, 0.89, 0.6,  0.14, 0.41, 0.9,  0.94, 0.51, 0.07, 0.92, 0.09, 0.59, 0.97, 0.32, 0.37, 0.98, 0.54, 0.43]
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
    plt.imshow(image.permute(1,2,0))
    plt.axis('off')
    print(label)

images = torch.cat(image_list, dim=0)
labels = torch.cat(label_list, dim=0)
true_labels = labels.detach().cpu().numpy().astype(np.int32)
print(true_labels)


def predict(x):
    tmp = torch.tensor(x).to(device)
    tmp = tmp.permute(0,3,1,2)
    output = model(tmp)
    return torch.sigmoid(output)


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
plt.figure(figsize=(10, 5))
for i in range(original_images.shape[0]):
    plt.subplot(1, original_images.shape[0], i + 1)
    plt.imshow(original_images[i])
    plt.axis('off')
    plt.title(f'Image {i + 1}')


output = predict(images)
print(output)

thresholds = [0.5]*num_labels
outputs = output.detach().cpu().numpy()
outputs_thresholded = (outputs > thresholds).astype(np.int32)
print(outputs_thresholded)


from sklearn.metrics import accuracy_score, confusion_matrix
acc_list, spec_list = [], []
labels_true = labels.detach().cpu().numpy().astype(np.int32)
for label_index in range(labels.shape[1]):
            label_true = labels_true[:, label_index]
            label_pred = outputs_thresholded[:, label_index]

            # Check if label_true and label_pred contain only one unique value
            if len(set(label_true)) == 1 and len(set(label_pred)) == 1:
                acc = accuracy_score(label_true, label_pred)
                spec = 0.0  # Set specificity to 0 when there's only one class present
            else:
                acc = accuracy_score(label_true, label_pred)
                tn, fp, _, _ = confusion_matrix(label_true, label_pred).ravel()
                spec = tn / (tn + fp) if (tn + fp) != 0 else 0.0

            acc_list.append(acc)
            spec_list.append(spec)

print(acc_list,"\n", spec_list)


# Initialize Shapley Explainer
masker = shap.maskers.Image("blur(64,64)", shape=images[0].shape)
explainer = shap.Explainer(predict, masker, output_names=class_names)

print("Type of explainer:", type(explainer))

shap_values = explainer(images, max_evals=10000, batch_size=50, outputs=shap.Explanation.argsort.flip[:2])
torch.cuda.empty_cache()

# Assuming shap_values contains your SHAP values
shap_data = inv_transform(shap_values.data).cpu().numpy()
shap_val = [val for val in np.moveaxis(shap_values.values, -1, 0)]
print(shap_data, shap_val)


def save_shap_image(shap_values, shap_data, labels, filename):
    shap.image_plot(shap_values=shap_values,pixel_values=shap_data,labels=labels,show=False)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()

save_shap_image(shap_val, shap_data, labels = shap_values.output_names, filename = filename1)

with open(filename2, 'w') as f:
    for i in range(len(idx)):
        sorted_output = torch.sort(output[i], descending=True)
        top_preds = np.array(sorted_output[0][:4].detach().cpu().numpy()) 
        top_indices = sorted_output[1][:4].cpu().numpy()
        formatted_preds = ', '.join([f'{pred:.4f}' for pred in top_preds])
        pred_classes = top_preds > 0.5
        top_classes = np.where(labels[i] == 1)[0]

        print(f'Image No. {idx[i]}, Image name {image_names[i]}:\nTop Predictions: {formatted_preds}\nTop Class IDs: {top_indices}'
            f'\nTop Class Names: {[class_names[idx] for idx in top_indices]}\nPredicted Classes: {[class_names[idx] for idx in top_indices[pred_classes]]}'
            f'\nTrue Classes: {[class_names[idx] for idx in top_classes]}\n')

        f.write(f'Image No. {idx[i]}:\n')
        f.write(f'Image names {image_names[i]}:\n')
        f.write(f'Top Predictions: {formatted_preds}\n')
        f.write(f'Top Class IDs: {top_indices}\n')
        f.write(f'Top Class Names: {[class_names[idx] for idx in top_indices]}\n')
        f.write(f'Predicted Classes: {[class_names[idx] for idx in top_indices[pred_classes]]}\n')
        f.write(f'True Classes: {[class_names[idx] for idx in top_classes]}\n\n')

    f.write('Thresholds:\n')
    f.write('\n'.join(map(str, threshold)) + '\n')
    
    f.write('\nClass Names:\n')
    f.write('\n'.join(class_names) + '\n')