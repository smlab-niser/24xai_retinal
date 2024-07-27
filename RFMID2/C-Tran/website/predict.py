import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
device = torch.device("cuda:0")
    
class result:
    def __init__(self, image_list, label_list, model):
        self.model = model
        self.images = torch.cat(image_list, dim=0)
        self.labels = torch.cat(label_list, dim=0)
        print(self.true_labels)
        print(self.labels.shape, self.images.shape)

    def predict(self, x):
        tmp = torch.tensor(x).to(device)
        tmp = tmp.permute(0,3,1,2)
        return torch.sigmoid(self.model(tmp))

    def prediction(self, thresholds):
        output = self.predict(self.images)
        outputs = output.detach().cpu().numpy()
        outputs_thresholded = (outputs > thresholds).astype(np.int32)

        acc_list, spec_list = [], []
        
        if self.labels is not None:
            labels_true = self.labels.detach().cpu().numpy().astype(np.int32)
            
            for label_index in range(self.labels.shape[1]):
                    label_true = labels_true[:, label_index]
                    label_pred = outputs_thresholded[:, label_index]

                    if len(set(label_true)) == 1 and len(set(label_pred)) == 1:
                        acc = accuracy_score(label_true, label_pred)
                        spec = 0.0  
                    else:
                        acc = accuracy_score(label_true, label_pred)
                        tn, fp, _, _ = confusion_matrix(label_true, label_pred).ravel()
                        spec = tn / (tn + fp) if (tn + fp) != 0 else 0.0

                    acc_list.append(acc)
                    spec_list.append(spec)

        return acc_list, spec_list, outputs
