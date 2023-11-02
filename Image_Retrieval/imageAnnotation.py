import cv2 as cv
import os
import shutil
import keyboard
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects
import tkinter as tk
from tkinter import filedialog

# Create a Tkinter window for folder selection.
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title="Select image source for annotation.")
annotation_folder = os.path.join(os.path.dirname(folder_path), "annotated")

weapon = False
vehicle = False
soldier = False
closeup = False
not_interesting = False

# Define a dictionary to map keys to variables
key_class_pairs = {
    "1": "weapon",
    "2": "vehicle",
    "3": "soldier",
    "0": "not_interesting",
    "4": "closeup"
}

annotations = {}



def toggle_variable_state(key):
    global weapon, vehicle, soldier, closeup, not_interesting
    variable_name = key_class_pairs.get(key)

    if variable_name:
        current_state = globals()[variable_name]
        new_state = not current_state
        globals()[variable_name] = new_state
        print(weapon, vehicle, soldier, closeup, not_interesting)

        # Update the text color based on the new state
        text_color = 'steelblue' if new_state else 'darkgrey'

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
            initial_text_color = 'steelblue' if globals()[variable_name] else 'darkgrey'

            annotation = plt.text(left_offset, y_position, class_name, fontsize=15, style = "oblique",
                                  color=initial_text_color, verticalalignment='center')
            annotation.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
            if key not in annotations:
                annotations[key] = []
            annotations[key].append(annotation)

        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.axis('off')  # Turn off axis labels
        plt.autoscale()
        plt.show()

        # After selection, track current variable status and append to list.
        true_classes = []
        if weapon:
            true_classes.append("weapon")
            shutil.copy(file, os.path.join(annotation_folder, "weapon"))
        if vehicle:
            true_classes.append("vehicle")
            shutil.copy(file, os.path.join(annotation_folder, "vehicle"))
        if soldier:
            true_classes.append("soldier")
            shutil.copy(file, os.path.join(annotation_folder, "soldier"))
        if closeup:
            true_classes.append("closeup")
            shutil.copy(file, os.path.join(annotation_folder, "closeup"))
        if not_interesting:
            true_classes.append("not_interesting")
            shutil.copy(file, os.path.join(annotation_folder, "not_interesting"))

        shutil.copy(file, os.path.join(annotation_folder, "all"))

        # print(true_classes)

        # Write class selection to file.
        with open(os.path.dirname(folder_path) + "/annotation.csv", "a", encoding = "utf-8") as csv_file:
            csv_file.write(filename + "," + ",".join(true_classes) + "\n")
            
        os.remove(file)
        

        # Reset variables
        weapon = False
        vehicle = False
        soldier = False
        closeup = False
        not_interesting = False

        # Unhook key presses
        for key in key_class_pairs.keys():
            keyboard.unhook_key(key)
