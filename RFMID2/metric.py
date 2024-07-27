from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, accuracy_score, confusion_matrix
import torch
import numpy as np

class Metric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.outputs = []
        self.labels = []
        self.outputs_thresholded =[]

    def update(self, outputs, labels, thresholds=None):
        
        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy().astype(np.int32)
    
        self.total_samples += outputs.shape[0]
        self.outputs.append(outputs)
        self.labels.append(labels)
    
    def calculate_thresholds(self, outputs, labels):
        thresholds = []

        for label_index in range(labels.shape[1]):
            label_outputs = outputs[:, label_index]
            label_true = labels[:, label_index]

            # Use F1 score to find the optimal threshold for each label
            f1_scores = [f1_score(label_true, (label_outputs > t).astype(np.int32), average='macro', zero_division=1) for t in np.arange(0, 1, 0.01)]
            optimal_threshold = np.arange(0, 1, 0.01)[np.argmax(f1_scores)]
            thresholds.append(optimal_threshold)

        return np.array(thresholds)

    def compute(self, thresholds = None):
        outputs = np.concatenate(self.outputs)
        labels = np.concatenate(self.labels)
        
        if thresholds is None: thresholds = self.calculate_thresholds(outputs, labels)
        outputs_thresholded = (outputs > thresholds).astype(np.int32)

        f1 = f1_score(labels, outputs_thresholded, average='macro', zero_division=1)
        mAP = average_precision_score(labels, outputs)
        AUC = roc_auc_score(labels, outputs, average='macro', multi_class='ovr')
 
        # Exclude NORMAL class
        ml_mAP = average_precision_score(labels[:, [0] + list(range(2, labels.shape[1]))], outputs[:, [0] + list(range(2, outputs.shape[1]))])
        ml_AUC = roc_auc_score(labels[:, [0] + list(range(2, labels.shape[1]))], outputs[:, [0] + list(range(2, outputs.shape[1]))]
                               , average='macro', multi_class='ovr')
        ml_score = (ml_mAP + ml_AUC) / 2

        # Include NORMAL class
        bin_AUC = roc_auc_score(labels[1], outputs[1])
        model_score = (ml_score + bin_AUC) / 2

        # Compute F1-score of NORMAL label
        bin_f1 = f1_score(labels[1], outputs_thresholded[1], average='binary', zero_division=1)

        # Calculate accuracy and specificity for each label
        acc_list = []
        spec_list = []
        for label_index in range(labels.shape[1]):
            label_true = labels[:, label_index]
            label_pred = outputs_thresholded[:, label_index]

            # Calculate accuracy
            acc = accuracy_score(label_true, label_pred)

            # Calculate specificity
            tn, fp, _, _ = confusion_matrix(label_true, label_pred).ravel()
            spec = tn / (tn + fp) if (tn + fp) != 0 else 0.0

            acc_list.append(acc)
            spec_list.append(spec)

        return f1, mAP, AUC, ml_mAP, ml_AUC, ml_score, bin_AUC, model_score, bin_f1, acc_list, spec_list, thresholds
     

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