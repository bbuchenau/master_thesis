import cv2 as cv
import os
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Create a Tkinter window for folder selection
root = tk.Tk()
root.withdraw()  # Hide the main window

# Ask the user to select the image source folder
folder_path = filedialog.askdirectory(title="Select the image source folder for annotation")

classes = ["gun", "rifle", "tank"]

# Check if the user canceled the folder selection
if not folder_path:
    print("Folder selection canceled.")
else:
    for filename in os.listdir(folder_path):
        file = os.path.join(folder_path, filename)

        image = cv.imread(file)
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.axis('off')  # Turn off axis labels
        plt.show()
        # TODO: Track keystroke events here and append to csv.
