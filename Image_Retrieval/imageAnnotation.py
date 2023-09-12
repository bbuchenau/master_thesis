import cv2 as cv
import os
import keyboard
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Create a Tkinter window for folder selection.
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title="Select image source for annotation.")

gun = False
rifle = False
tank = False

# Define a dictionary to map keys to variables
key_class_pairs = {
    "1": "gun",
    "2": "rifle",
    "3": "tank"
}

annotations = {}

def toggle_variable_state(key):
    global gun, rifle, tank
    variable_name = key_class_pairs.get(key)

    if variable_name:
        current_state = globals()[variable_name]
        new_state = not current_state
        globals()[variable_name] = new_state
        print(gun, rifle, tank)

        # Update the text color based on the new state
        text_color = 'blue' if new_state else 'grey'

        # Update the text color on the existing plot
        for annotation in annotations[key]:
            annotation.set_color(text_color)

        # Redraw the plot to apply the color change
        plt.draw()

# Check if the user canceled the folder selection
if not folder_path:
    print("Folder selection canceled.")
else:
    for filename in os.listdir(folder_path):

        for key in key_class_pairs.keys():
            keyboard.on_press_key(key, lambda event, key=key: toggle_variable_state(key))

        file = os.path.join(folder_path, filename)
        image = cv.imread(file)

        # Set the figure size to 800x600 pixels
        plt.figure(figsize=(10, 6))

        left_offset = 0
        annotations.clear()  # Clear annotations for each image

        # Add annotations for each class name below each other on the left side
        for key, variable_name in key_class_pairs.items():
            y_position = (int(key) - .5) * 15  # Adjust this value for vertical spacing
            class_name = key_class_pairs[key]

            # Determine the initial text color based on the initial state
            initial_text_color = 'blue' if globals()[variable_name] else 'grey'

            annotation = plt.text(left_offset, y_position, class_name, fontsize=15, style = "oblique",
                                  color=initial_text_color, verticalalignment='center')
            if key not in annotations:
                annotations[key] = []
            annotations[key].append(annotation)

        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.axis('off')  # Turn off axis labels
        plt.autoscale()
        plt.show()

        # Reset variables
        gun = False
        rifle = False
        tank = False

        # Unhook key presses
        for key in key_class_pairs.keys():
            keyboard.unhook_key(key)
