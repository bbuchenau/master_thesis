### This script looks through project directories from the SALW project and extracts the images. ###

import os
import shutil
from tkinter import filedialog

# Create Tkinter window for folder selection, as this is applied on multiple datasets.
folder_path = filedialog.askdirectory(title="Original data folder.")
destination_path = filedialog.askdirectory(title="Source data folder.")

# Set counter for new naming.
count = 0

# Iterate through all subdirectories that contain images.
for weapon_folder in os.listdir(folder_path):
    weapon_folder_path = os.path.join(folder_path, weapon_folder)
    for file in os.listdir(weapon_folder_path):
        # Only extract files that are valid images.
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            count += 1
            new_filename = f"image_{count}.png"
            # Copy to destination folder.
            shutil.copy(os.path.join(weapon_folder_path, file), os.path.join(destination_path, new_filename))
