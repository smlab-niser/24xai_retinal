import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

# This class generates the dataset
class RetinaDataset(Dataset):
    
    def __init__(self, data_dir, split, omit, transform=None):  
        self.data_dir = data_dir                           # Initialize the directory for data
        self.transform = transform                         # Initialize the transformation parameter
        self.image_paths = []                              # Initialize the list to store paths to the images
        self.labels = []                                   # Initialize the list to store the label vector of each sample
        
        csv_path = os.path.join(data_dir, './set1', f'{split}.csv') # stores the path to the label csv file
        print(csv_path)
        data = pd.read_csv(csv_path)                                           # data stores the label dataframe
        
        for idx, row in data.iterrows():   # idx stores the column index and row stores the row for each sample
            
            if 'mured' in omit:
                if row["ID"].startswith("aria") or row["ID"].startswith("(") or row["ID"].startswith("im") or row["ID"].isdigit(): continue  
            if 'rfmid2' in omit:
                if row["ID"].startswith("rf"): continue 
            if 'jseic' in omit:
                if row["ID"].startswith("js"): continue  # Skip the row if ID starts with "js"
            
            if os.path.exists(os.path.join(data_dir, 'images', f'{row["ID"]}.tif')): # check the extension of the image to add to the image path
                image_path = os.path.join(data_dir, 'images', f'{row["ID"]}.tif') 
            elif os.path.exists(os.path.join(data_dir, 'images', f'{row["ID"]}.png')): 
                image_path = os.path.join(data_dir, 'images', f'{row["ID"]}.png')
            else:
                image_path = os.path.join(data_dir, 'images', f'{row["ID"]}.jpg')
                
            self.image_paths.append(image_path)                                      # Appending paths to image_paths
            self.labels.append(row[1:].tolist())                                     # Appending label vectors to labels
    
    def __len__(self):                                                        
        return len(self.image_paths)
    
    # This method returns a sample image with applied transformation and the PyTorch tensor for the labels
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)                                     # To open an image from the path
        
        if self.transform:                                                 # Applying transformation to image
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)        # Generating label tensor
        
        return image, label                                                
    
    def get_grayscale_images(self):
        grayscale_images = []
        for image_path in self.image_paths:
            image = Image.open(image_path)
            if len(image.getbands()) == 1:  # Check if the image has only one channel (grayscale)
                grayscale_images.append(os.path.basename(image_path))
        return grayscale_images