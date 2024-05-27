import sys
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import math

def extract_palette(image_path, num_colors=5):
     
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
     fig, ax = plt.subplots(figsize=(5,2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
     hex_strings = []
     
     # Display the palette
     for i, color in enumerate(colors):
          hex_string = '#%02x%02x%02x' % tuple(color)
          ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=hex_string))
          hex_strings.append(hex_string)
          
     # Adjust the plot limits and show the palette
     ax.set_xlim(0, len(colors))
     ax.set_ylim(0, 1)
     plt.show()
     
     return hex_strings

def get_closest_color(hex_strings):
     
     colors = [
          '#fc0000', # Red
          '#ff7b00', # Orange
          '#ffff00', # Yellow
          '#7bff00', # Yellow Green
          '#00ff00', # Green
          '#00ff7d', # Turquoise
          '#00ffff', # Cyan
          '#007bff', # Ocean
          '#0000ff', # Blue
          '#7b00ff', # Violet
          '#ff00ff', # Magenta
          '#ff007b' # Raspberry
     ]
     
     closest_colors = []

     for hex_string in hex_strings:
          input_color = tuple(int(hex_string.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
          min_distance = float('inf')
          closest_hex = None

          for color in colors:
               base_color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
               distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(base_color, input_color)]))
               
               if distance < min_distance:
                    min_distance = distance
                    closest_hex = color

          closest_colors.append(closest_hex)
          print(f"Closest color to {hex_string} is {closest_hex}")

     return closest_colors
          
def main():
     
     colors = extract_palette('samples/sample3.jpg')
     hex = display_palette(colors)
     
     get_closest_color(hex)
          
if __name__ == "__main__":
     main()
