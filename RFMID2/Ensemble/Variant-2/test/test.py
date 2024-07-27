import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import sys
sys.path.append("..")
sys.path.extend(["../..", "../../../backbone","../../..", ".."])
from dataloader import create_dataloader
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
torch.cuda.empty_cache()

# Set device to GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {torch.cuda.get_device_name(torch.cuda.current_device())}" if torch.cuda.is_available() else "Current device: CPU")

# Define hyperparameters
batch_size = 16
image_size = 384
num_workers = 8
num_classes = 21
data_dir = '../../../../../data/GT-main'
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

with open('./results/1.txt', 'w') as f:
    for n, omit in enumerate(omission):
        # Create test dataloader
        test_dataloader = create_dataloader(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, size=image_size, phase='test', omit=omit)

        if n == 0:
            thresholds = [0.5, 0.27, 0.2, 0.68, 0.69, 0.51, 0.54, 0.32, 0.59, 0.91, 0.72, 0.29, 0.32, 0.54, 0.79, 0.34, 0.95, 0.62, 0.46, 0.54, 0.09]
            model = torch.load('../model/en2.pth', map_location=device)
        elif n == 1:
            thresholds = [0.18, 0.5, 0.73 ,0.42, 0.44, 0.27, 0.51, 0.54, 0.74, 0.26, 0.62, 0.06, 0.62, 0.13, 0.73, 0.93, 0.26, 0.26, 0.22, 0.33, 0.03]
            model = torch.load('../model/en2-1.pth', map_location=device)
        else:
            thresholds = [0.25, 0.39, 0.38, 0.76, 0.78, 0.48, 0.88, 0.23, 0.36, 0.93, 0.64, 0.15, 0.45, 0.84, 0.09, 0.72, 0.39, 0.4, 0.03, 0.43, 0.08]
            model = torch.load('../model/en2-js.pth', map_location=device)
        
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


    