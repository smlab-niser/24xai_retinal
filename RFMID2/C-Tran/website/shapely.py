import numpy as np
import torch
import sys
sys.path.append("..", "../backbone")
import torchvision.transforms as transforms
import shap

device = torch.device("cuda:0")
print(f"Current device: {torch.cuda.get_device_name(torch.cuda.current_device())}" if torch.cuda.is_available() else "Current device: CPU")

num_labels = 21
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
class_names = ["DR","NORMAL","MH","ODC","TSLN","ARMD","DN","MYA","BRVO","ODP","CRVO","CNV","RS","ODE","LS","CSR","HTR","ASR","CRS","OTHER","RB"]


inv_transform = transforms.Compose([
transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
transforms.Normalize(mean=(-1 * np.array(mean) / np.array(std)).tolist(),std=(1 / np.array(std)).tolist(),),
transforms.Lambda(lambda x: x.permute(0, 2, 3, 1)),
])

class predictor:
    def __init__(self, model):
        self.model = model      
        
    def predict(self, x):
        tmp = torch.tensor(x).to(device)
        tmp = tmp.permute(0,3,1,2)
        return torch.sigmoid(self.model(tmp))

class masker:
    def __init__(self, images, model, output, idx=[0], size=2):
        self.images = images
        self.size = size
        self.idx = idx
        
        self.masker = shap.maskers.Image("blur(64,64)", shape=images[0].shape)
        test = predictor(model)  
        self.explainer = shap.Explainer(test.predict, self.masker, output_names=class_names)
        self.output = output
        
    def shapley(self, labels=None, max_evals=10000, bs=50, size=2):
        shap_values = self.explainer(self.images, mx_evals=max_evals, batch_size=bs, outputs=shap.Explanation.argsort.flip[:size])
        shap_data = inv_transform(shap_values.data).cpu().numpy()
        shap_val = [val for val in np.moveaxis(shap_values.values, -1, 0)]

        image_plot = shap.image_plot(shap_values=shap_val, pixel_values=shap_data, labels=shap_values.output_names)

        for i in range(len(self.idx)):
            
            sorted_output = torch.sort(self.output[i], descending=True)
            top_preds = np.array(sorted_output[0][:size].detach().cpu().numpy()) 
            top_indices = sorted_output[1][:size].cpu().numpy()
            formatted_preds = ', '.join([f'{pred:.4f}' for pred in top_preds])
            pred_classes = top_preds>0.5
            
            if labels is None: top_classes = np.where(labels[i] == 1)[0]
            else: top_classes = []

            info_string = (f'Image No. {self.idx[i]}:\n'
                   f'Top Predictions: {formatted_preds}\n'
                   f'Top Class IDs: {top_indices}\n'
                   f'Top Class Names: {[class_names[idx] for idx in top_indices]}\n'
                   f'Predicted Classes: {[class_names[idx] for idx in top_indices[pred_classes]]}\n'
                   f'True Classes: {[class_names[idx] for idx in top_classes]}\n')
            
            return info_string, image_plot