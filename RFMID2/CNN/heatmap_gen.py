import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def generate_hm(image_names, heatmaps, save_loc):

    for i, im_name in enumerate(image_names):

        image_path = '../../../data/GT-main/images/'+im_name  
        image = cv2.imread(image_path)
        image = cv2.resize(image, (384, 384))

        # Resize the heatmap to match the image dimensions
        heatmap_resized = cv2.resize(heatmaps[i], (image.shape[1], image.shape[0]))

        # Normalize the heatmap to the range [0, 1]
        heatmap_resized = np.float32(heatmap_resized)
        heatmap_resized = heatmap_resized - np.min(heatmap_resized)
        heatmap_resized = heatmap_resized / np.max(heatmap_resized)

        # Apply a color map to the heatmap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_RAINBOW)
        heatmap_colored = np.float32(heatmap_colored) / 255

        # Overlay the heatmap on the image
        overlayed_image = heatmap_colored + np.float32(image) / 255
        overlayed_image = overlayed_image / np.max(overlayed_image)

        # Display the resulting image
        plt.imshow(overlayed_image)
        plt.axis('off')  # Hide axis
        plt.show()

        # Save the plot to the specified location with the name im_name.png
        save_path = os.path.join(save_loc, im_name.split('.')[0] + '_heatmap.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the plot to avoid displaying it in interactive environments
