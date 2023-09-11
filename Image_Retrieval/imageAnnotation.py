import cv2 as cv
import os
import keyboard
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Create a Tkinter window for folder selection
root = tk.Tk()
root.withdraw()  # Hide the main window

# Ask the user to select the image source folder
folder_path = filedialog.askdirectory(title="Select the image source folder for annotation")

gun = False
rifle = False
tank = False

# Define a dictionary to map keys to variables
key_class_pairs = {
    "1": "gun",
    "2": "rifle",
    "3": "tank"
}

def toggle_variable_state(key):
    global gun, rifle, tank
    variable_name = key_class_pairs.get(key)
    
    if variable_name:
        current_state = globals()[variable_name]
        globals()[variable_name] = not current_state
        print(gun, rifle, tank)

# Check if the user canceled the folder selection
if not folder_path:
    print("Folder selection canceled.")
else:
    for filename in os.listdir(folder_path):
        
        for key in key_class_pairs.keys():
            keyboard.on_press_key(key, lambda event, key=key: toggle_variable_state(key))
        
        file = os.path.join(folder_path, filename)
        image = cv.imread(file)
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.axis('off')  # Turn off axis labels
        plt.show()

        # Print the variables
        print("gun:", gun)
        print("rifle:", rifle)
        print("tank:", tank)

        # Reset variables
        gun = False
        rifle = False
        tank = False

        # Unhook key presses
        for key in key_class_pairs.keys():
            keyboard.unhook_key(key)