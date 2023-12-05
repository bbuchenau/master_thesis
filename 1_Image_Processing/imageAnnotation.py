### This script iterates over a selected image dataset and enables quick annotation with all the specified ###
### classes using keystrokes. It creates a csv file with all records and moves the images to class folders. ###

# TODO: Create config file for all settings in the script!

import cv2 as cv
import os
import json
import inspect
import shutil
import keyboard
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects
import tkinter as tk
from tkinter import filedialog

# Create a Tkinter window for folder selection, as this might be applied to multiple datasets.
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title="Select image source for annotation.")
annotation_folder = os.path.join(os.path.dirname(folder_path), "annotated")

# Set script path as current dir and load annotationConfig JSON file that stores key/value pairs.
# Key in this case actually means the pressed keystroke and value means the model class.
current_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
config_filepath = os.path.join(current_directory, "annotationConfig.json")
with open(config_filepath, "r") as jsonfile:
    key_class_pairs = json.load(jsonfile)

# Create binary variable for each value (class) that keeps track of the state, and set to False.
for klasse in key_class_pairs.values():
    globals()[klasse] = False

# Save selected annotations here.
annotations = {}

# Define function that updates the variable (class) states between True and False.
def toggle_variable_state(key):
    global weapon, vehicle, soldier, interesting, not_interesting
    variable_name = key_class_pairs.get(key)

    if variable_name:
        current_state = globals()[variable_name]
        new_state = not current_state
        globals()[variable_name] = new_state
        # Print for testing.
        #print(weapon, vehicle, soldier, interesting, not_interesting)

        # Update text color based on the new state.
        text_color = 'steelblue' if new_state else 'darkgrey'

        # Update text color on the existing plot.
        for annotation in annotations[key]:
            annotation.set_color(text_color)

        # Redraw plot to apply color change.
        plt.draw()

# Iterate through images in the folder.
if folder_path:
    for filename in os.listdir(folder_path):

        for key in key_class_pairs.keys():
            keyboard.on_press_key(key, lambda event, key=key: toggle_variable_state(key))

        file = os.path.join(folder_path, filename)
        image = cv.imread(file)

        # Set the figure size to 800x600 pixels.
        plt.figure(figsize=(10, 6))

        left_offset = 0
        # Clear annotations for each image.
        annotations.clear()  

        # Add annotations for each class name below each other on the left side.
        # They are in the image itself, so I reduce my eye movement for comfort.
        for key, variable_name in key_class_pairs.items():
            # Vertical spacing.
            y_position = (int(key) - .5) * 15
            class_name = key_class_pairs[key]

            # Determine initial text color based on the initial state, which is always grey first.
            initial_text_color = 'steelblue' if globals()[variable_name] else 'darkgrey'

            # Modify text appearance, settings right now make most sense for optimal annotation.
            annotation = plt.text(left_offset, y_position, class_name, fontsize=15, style = "oblique",
                                  color=initial_text_color, verticalalignment='center')
            annotation.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
            if key not in annotations:
                annotations[key] = []
            annotations[key].append(annotation)

        # Display image.
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.axis('off')
        plt.autoscale()

        # Get the figure manager and set image position (my manual solution for the screen). 
        manager = plt.get_current_fig_manager()
        manager.window.wm_geometry("+2400+400") #first x then y.
        plt.show()

        # After selection, track current variable status and if true
        # append to list and copy file to new class folder.
        true_classes = []
        for klasse in key_class_pairs.values():
            if globals()[klasse]:
                true_classes.append(f"{klasse}")
                shutil.copy(file, os.path.join(annotation_folder, f"{klasse}"))

        shutil.copy(file, os.path.join(annotation_folder, "all"))

        # Print for testing.
        # print(true_classes)

        # Write class selection to file.
        with open(os.path.dirname(folder_path) + "/annotation.csv", "a", encoding = "utf-8") as csv_file:
            csv_file.write(filename + "," + ",".join(true_classes) + "\n")

        # Remove "old" file from ogirinal image folder, as it is copied before.
        os.remove(file)
        
        # Reset class variables to False again.
        for klasse in key_class_pairs.values():
            globals()[klasse] = False

        # Unhook key presses
        for key in key_class_pairs.keys():
            keyboard.unhook_key(key)
