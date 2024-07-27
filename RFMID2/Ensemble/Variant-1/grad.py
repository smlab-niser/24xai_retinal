import torch 
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import numpy as np

def apply_grad_cam(model, input_image, gradients, class_idx, target_layer):

    # Forward pass
    output = model(input_image)

    # Zero gradients
    model.zero_grad()

    # Backward pass
    output[:, class_idx].backward()

    # Get the gradients and feature map
    grad = gradients[0]  # assuming batch size of 1
    feature_map = target_layer.output.detach()

    # Compute the weights
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)

    # Compute the Grad-CAM heatmap
    grad_cam = torch.sum(weights * feature_map, dim=1).squeeze().relu()
    grad_cam = F.interpolate(grad_cam.unsqueeze(0), size=input_image.shape[2:], mode='bilinear', align_corners=False)
    grad_cam = grad_cam.squeeze().cpu().numpy()

    # Normalize the heatmap
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())

    # Visualize the heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # Convert BGR to RGB
    superimposed_img = heatmap + np.float32(input_image.squeeze().permute(1, 2, 0).cpu().numpy())
    superimposed_img = superimposed_img / np.max(superimposed_img)

    plt.imshow(superimposed_img)
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.show()