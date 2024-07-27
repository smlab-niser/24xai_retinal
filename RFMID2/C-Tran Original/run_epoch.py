import argparse,math,time,warnings,copy, numpy as np, os.path as path
from utils.assymetric_loss_opt import AsymmetricLossOptimized 
import torch, torch.nn as nn, torch.nn.functional as F
from pdb import set_trace as stop
from tqdm import tqdm
from models.utils import custom_replace
import random

from utils.poly_loss import BCEPolyLoss, FLPolyLoss

grad_ac_steps = 1
num_labels = 21
batch_size =  16

def get_class_weights(y_true):
    y_pos = np.sum(y_true, axis=0)
    weights = y_pos.max() / y_pos

    return torch.Tensor(weights)


def run_epoch(model,data,optimizer,epoch,desc,device,metric,train=False,warmup_scheduler=None):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    # pre-allocate full prediction and target tensors
    all_predictions = []
    all_targets = []
    all_masks = []

    max_samples = -1

    batch_idx = 0
    loss_total = 0
    unk_loss_total = 0

    criterion = nn.BCEWithLogitsLoss(weight=get_class_weights(data.dataset.get_labels()).to(device), reduction='none')

    for batch in tqdm(data,mininterval=0.5,desc=desc,leave=False,ncols=50):
        if batch_idx == max_samples:
            break

        labels = batch['labels'].float()
        images = batch['image'].float()
        mask = batch['mask'].float()
        unk_mask = custom_replace(mask,1,0,0)
        
        mask_in = mask.clone()
        
        if train:
            pred,int_pred,attns = model(images.to(device),mask_in.to(device))
        else:
            with torch.no_grad():
                pred,int_pred,attns = model(images.to(device),mask_in.to(device))
                
        # if batch_idx == 0: print(images.shape, labels.shape, pred.shape)
    
        loss = criterion(pred.view(labels.size(0),-1), labels.to(device))
        
        loss_out = loss.sum() 

        if train:
            loss_out.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if warmup_scheduler is not None:
                warmup_scheduler.step()
                    
        with torch.no_grad(): metric.update(pred, labels)

        ## Updates ##
        loss_total += loss_out.item()
        # start_idx,end_idx=(batch_idx*batch_size),((batch_idx+1)*batch_size)
        
        # if pred.size(0) != all_predictions[start_idx:end_idx].size(0):
        #     pred = pred.view(labels.size(0),-1)
        
        # all_predictions[start_idx:end_idx] = pred.data.cpu()
        # all_targets[start_idx:end_idx] = labels.data.cpu()
        # all_masks[start_idx:end_idx] = mask.data.cpu()
        all_predictions += pred.data.cpu().tolist()
        all_targets += labels.data.cpu().tolist()
        all_masks += mask.data.cpu().tolist()
        
    loss_total = loss_total/float(len(all_predictions))

    return all_predictions,all_targets,all_masks,loss_total