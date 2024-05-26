import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 

def extract_palette(image_path, num_colors=10):
     
     # Load image and convert to RGB
     image = cv2.imread(image_path)
     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     image = image.reshape((image.shape[0] * image.shape[1], 3))
     
     # Apply KMeans clustering
     kmeans = KMeans(n_clusters=num_colors)
     kmeans.fit(image)
     
     # Get the colors 
     colors = kmeans.cluster_centers_.astype(int)
     
     return colors

def display_palette(colors):
     # Create a figure and axes
     fig, ax = plt.subplots(figsize=(10,2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
     
     # Display the palette
     for i, color in enumerate(colors):
          hex_string = '#%02x%02x%02x' % tuple(color.reshape(1, -1)[0])
          ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=hex_string))
          
     # Adjust the plot limits and show the palette
     ax.set_xlim(0, len(colors))
     ax.set_ylim(0, 1)
     plt.show()

def main():
     
     colors = extract_palette('sample2.jpg')
     display_palette(colors)
          
if __name__ == "__main__":
     main()
