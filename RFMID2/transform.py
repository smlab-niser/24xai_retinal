import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class Transform:
    def __init__(self, size, phase):
        self.phase = phase
        print(size)
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.Lambda(lambda x: x.convert("RGB") if x.mode == 'P' else x),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert to 3 channels if grayscale
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((size, size)),
                transforms.Lambda(lambda x: x.convert("RGB") if x.mode == 'P' else x),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert to 3 channels if grayscale
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((size, size)),
                transforms.Lambda(lambda x: x.convert("RGB") if x.mode == 'P' else x),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert to 3 channels if grayscale
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'all': transforms.Compose([
                transforms.Resize((size, size)),
                transforms.Lambda(lambda x: x.convert("RGB") if x.mode == 'P' else x),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert to 3 channels if grayscale
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        }

    def __call__(self, sample):
        try:
            if sample.mode == 'RGBA':
                sample = sample.convert('RGB')

            if isinstance(sample, Image.Image):
                transformed_image = self.data_transforms[self.phase](sample)
            elif isinstance(sample, (np.ndarray, torch.Tensor)):
                w, h = sample.size
                aspect_ratio = w / h
                new_h = sample.size[1]
                new_w = int(new_h * aspect_ratio)
                resized_image = sample.resize((new_w, new_h))
                transformed_image = self.data_transforms[self.phase](resized_image)
            else:
                raise TypeError(f"Input should be Tensor, ndarray, or PIL Image. Got {type(sample)}.")

        except Exception as e:
            print(f"Error in transformation - {e}")
            print(f"Image mode before transformation: {sample.mode}")
            print(f"Image details: {sample}")
            transformed_image = sample  # If an error occurs, return the original image

        return transformed_image
    