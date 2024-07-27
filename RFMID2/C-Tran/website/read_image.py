import os
from torchvision import transforms
from PIL import Image

def read_images(folder_path, image_size):
    # Define transformations to be applied to each image
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Lambda(lambda x: x.convert("RGB") if x.mode == 'P' else x),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert to 3 channels if grayscale
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_list = []

    for file_name in os.listdir(folder_path):
        # Check if the file is an image file
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            image_list.append(image)

    return image_list