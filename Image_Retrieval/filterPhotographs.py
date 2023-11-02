import os
import cv2 as cv
import numpy as np
from tkinter import filedialog
from matplotlib import pyplot as plt

similarity_threshold = 0.995
unique_colors_threshold = 100000

# Function to count unique colors in the images.
def count_unique_colors(image):
    unique_colors = np.unique(image.reshape(-1, image.shape[2]), axis=0)
    return unique_colors.shape[0]

# Create a Tkinter window for folder selection.
folder_path = filedialog.askdirectory(title="Select image source for photograph filter.")

# Iterate through files to perform the analysis.
for filename in os.listdir(folder_path):
    
    # Read and resize image.
    image_path = os.path.join(folder_path, filename)
    image = cv.imread(image_path)
    image = cv.resize(image, (1024, 1024))

    # Blurring the image for color gradient check.
    blurred = cv.bilateralFilter(image, 10, 250, 250)

    # Calculate the number of unique colors in the image.
    unique_colors_count = count_unique_colors(image)

    # Compare original and blurred image to see similarity indicating the gradient change.
    diffs = []
    for k, color in enumerate(('b', 'r', 'g')):
        real_histogram = cv.calcHist(image, [k], None, [256], [0, 256])
        color_histogram = cv.calcHist(blurred, [k], None, [256], [0, 256])
        diffs.append(cv.compareHist(real_histogram, color_histogram, cv.HISTCMP_CORREL))
    threshold = sum(diffs) / 3

    # Generate color histogram.
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv.calcHist([image],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

    # Identify (most likely) cartoon/animated images and delete them, keep all (assumed) photographs.
    if threshold > similarity_threshold and unique_colors_count < unique_colors_threshold:
        print(f"{filename} - Most likely animated, deleting! Colors: " + str(unique_colors_count) + " Similarity: " + str(threshold))
        #os.remove(image_path)
    else:
        print(f"{filename} - Photograph. Colors: " + str(unique_colors_count) + " Similarity: " + str(threshold))

cv.destroyAllWindows()
