import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import sys
sys.path.append("..")
sys.path.extend(["../..", "../../backbone","../../..", ".."])
from dataloader import create_dataloader
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
torch.cuda.empty_cache()

# Set device to GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {torch.cuda.get_device_name(torch.cuda.current_device())}" if torch.cuda.is_available() else "Current device: CPU")

# Define hyperparameters
batch_size = 16
num_epochs = 200
learning_rate = 0.000001
in_channel = 3
image_size = 384
patch_size = [32,16,16,16,16,8]
num_workers = 4
embed_dim = 960
mlp_dim = 1024
dim_feedforward = 1024
num_labels = 21
num_layers = 12
num_classes = 21
data_dir = '../../../../data/GT-main'
omission = ['None', '-rf', '-js']

def calculate_metrics(test_labels, test_preds, thresholds):
    num_classes = test_labels.shape[1]
    metrics_dict = {}
    thresholded_test_preds = np.where(test_preds > np.array(thresholds).reshape(1, -1), 1, 0)

    for label in range(num_classes):
        TP = np.sum((test_labels[:, label] == 1) & (thresholded_test_preds[:, label] == 1))
        FP = np.sum((test_labels[:, label] == 0) & (thresholded_test_preds[:, label] == 1))
        TN = np.sum((test_labels[:, label] == 0) & (thresholded_test_preds[:, label] == 0))
        FN = np.sum((test_labels[:, label] == 1) & (thresholded_test_preds[:, label] == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 1.0
        auc = roc_auc_score(test_labels[:, label], test_preds[:, label])

        metrics_dict[label] = {'Precision': precision, 'Recall': recall, 'F1': f1, 'AUC': auc}

    return metrics_dict

with open('./results/2.txt', 'w') as f:
    for n, omit in enumerate(omission):
        # Create test dataloader
        test_dataloader = create_dataloader(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, size=image_size, phase='test', omit=omit)

        if n == 0:
            # thresholds = [0.5, 0.7, 0.83, 0.75, 0.89, 0.6, 0.14, 0.41, 0.9, 0.94, 0.51, 0.07, 0.92, 0.09, 0.59, 0.97, 0.32, 0.37, 0.98, 0.54, 0.43]
            # model = torch.load('../models/iecte1.pth', map_location=device)
            thresholds = [0.46, 0.24, 0.61, 0.42, 0.15, 0.55, 0.73, 0.61, 0.28, 0.61, 0.3, 0.79, 0.85, 0.32, 0.87, 0.37, 0.99, 0.69, 0.08, 0.14, 0.99]
            model = torch.load('../models/ieect1.pth', map_location=device)
        elif n == 1:
            # thresholds = [0.05, 0.73, 0.25, 0.96, 0.53, 0.06, 0.43, 0.16, 0.87, 0.9, 0.95, 0.06, 0.01, 0.43, 0.42, 0.02, 0.6, 0.93, 0.5, 0.31, 0.01]
            # model = torch.load('../models/iecte2.pth', map_location=device)
            thresholds = [0.49, 0.86, 0.08, 0.57, 0.37, 0.2, 0.05, 0.02, 0.07, 0.95, 0.63, 0.01, 0.55, 0.01, 0.17, 0.06, 0.16, 0.01, 0.96, 0.14, 0.08]
            model = torch.load('../models/ieect2.pth', map_location=device)
        else:
            # thresholds = [0.85, 0.11, 0.05, 0.61, 0.58, 0.9, 0.65, 0.28, 0.11, 0.99, 0.68, 0.36, 0.26, 0.9, 0.11, 0.88, 0.07, 0.15, 0.22, 0.32, 0.04]
            # model = torch.load('../models/iecte-js.pth', map_location=device)
            thresholds = [0.52, 0.87, 0.05, 0.19, 0.61, 0.63, 0.08, 0.35, 0.54, 0.91, 0.01, 0.89, 0.51, 0.24, 0.15, 0.76, 0.99, 0.62, 0.9, 0.88, 0.01]
            model = torch.load('../models/ieect-js.pth', map_location=device)
        
        model.to(device)

        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(test_dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                test_preds += outputs.tolist()
                test_labels += labels.tolist()

        test_labels = np.array(test_labels)
        test_preds = np.array(test_preds)
        torch.cuda.empty_cache()

        metrics_dict = calculate_metrics(test_labels, test_preds, thresholds)

        # Print the dictionary in tabular format using the tabulate library
        headers = ['Label', 'Precision', 'Recall', 'F1', 'AUC']
        table = []
        for label in range(len(metrics_dict)):
            row = [label]
            for metric in ['Precision', 'Recall', 'F1', 'AUC']:
                value = metrics_dict[label][metric]
                row.append('{:.4f}'.format(value))
            table.append(row)

        # Convert the table to a string
        table_str = tabulate(table, headers=headers)
        print(table_str)

    
        f.write(f'{omit}\n{table_str}\n')
        f.flush()


    