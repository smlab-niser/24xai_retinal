import matplotlib.pyplot as plt
import json
import torch
import sys
sys.path.append("..")
sys.path.append("../backbone")
from shapely import masker
from predict import result
from read_image import read_images
torch.cuda.empty_cache()

device = torch.device("cuda:0")
model = torch.load('models/dn1.pth')
model.to(device)
print(f"Current device: {torch.cuda.get_device_name(torch.cuda.current_device())}" if torch.cuda.is_available() else "Current device: CPU")

data_dir = 'uploaded_image/'
image_size = 384
num_labels = 21
thresholds = [0.65, 0.7, 0.49, 0.43, 0.66, 0.78, 0.26, 0.21, 0.38, 0.82, 0.64, 0.9, 0.38, 0.59, 0.86, 0.31, 0.51, 0.69, 0.91, 0.23, 0.37]
thresholds = [0.5]*num_labels

image_list = read_images(data_dir, image_size)
label_list = []

# Function to perform prediction and masker
def execute_explanation(image_list, label_list, model, idx):
    
    images = torch.cat(image_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    test = result(image_list, label_list, labels, model)
    accuracy, specificity, output = test.prediction(thresholds)
    
    Masker = masker(images, model, output, idx=idx)
    info_string, image_plot = Masker.shapely()

    return accuracy, specificity, info_string, image_plot

# Main execution
if __name__ == "__main__":
    accuracy, specificity, info_string, image_plot = execute_explanation(image_list, label_list, model, idx=[0])

    # Serialize the objects into a dictionary
    result_dict = {
        "accuracy": accuracy,
        "specificity": specificity,
        "info_string": info_string,
        "image_plot": image_plot
    }

    # Write the dictionary to a JSON file
    with open('remote_result.json', 'w') as json_file:
        json.dump(result_dict, json_file)

    # Print confirmation message
    print("Result saved to remote_result.json")