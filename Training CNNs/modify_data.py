# Code to extract desired data from the preprovided datasets
import os
import shutil
from PIL import Image

# Extract white faces
# Path to the directory containing the dataset
dataset_path = '/home/sruj/Downloads/archive (1)/casia-webface'

# Path to the metadata file
metadata_path = '/home/sruj/Downloads/archive (1)/casia-webface.txt'

# Path to the output directory for the extracted images
output_dir = '/home/sruj/Downloads/archive (1)/white'

# List of ethnicity labels to include
white_ethnicities = [0]

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the metadata file into a dictionary
metadata_dict = {}
with open(metadata_path) as f:
    for line in f:
        fields = line.strip().split()
        image_path = os.path.join(dataset_path, fields[0])
        ethnicity = int(fields[1])
        metadata_dict[image_path] = ethnicity

# Iterate through the images and copy the images of white individuals to the output directory
for image_path in metadata_dict:
    ethnicity = metadata_dict[image_path]
    if ethnicity in white_ethnicities:
        # Copy the image to the output directory
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        shutil.copy(image_path, output_path)


# Invert image
# Set the path of the original folder
original_folder_path = "/home/sruj/Downloads/archive (2)/desired dataset"

# Set the path of the new folder where rotated images will be saved
rotated_folder_path = "/home/sruj/Downloads/archive (2)/inverted dataset"

# Create the rotated folder if it does not exist
if not os.path.exists(rotated_folder_path):
    os.makedirs(rotated_folder_path)

# Loop through all the images in the original folder
for filename in os.listdir(original_folder_path):
    # Open the image file
    img = Image.open(os.path.join(original_folder_path, filename))
    # Rotate the image by 180 degrees
    rotated_img = img.rotate(180)
    # Save the rotated image to the new folder
    rotated_img.save(os.path.join(rotated_folder_path, filename))

