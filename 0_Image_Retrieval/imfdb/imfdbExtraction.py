### This script combines the two separate processes of extracting the movie names and ###
### downloading the relevant movie images to a folder. Images are stored only locally! ###

import sys
import os
import re
import json
import requests
import inspect
from bs4 import *

########## Function to extract all the movie titles from IMFDB. ##########
def extract_movie_names(url):
    # Send GET request to URL.
    response = requests.get(url)

    # Parse HTML content to the "soup", then pin down on the
    # wanted tag, which are list items within the mw-pages tag.
    soup = BeautifulSoup(response.content, "html.parser")
    div_tag = soup.find("div", id = "mw-pages")
    list_tags = div_tag.find_all("li")

    # Store text of list items (which are the movie names) and return.
    movie_names = []
    for tag in list_tags:
        name = tag.text
        #--- Manipulate name string here if I want! ---
        movie_names.append(name)
    return movie_names

########## Function to save list to textfile. ########## 
def save_textfile(list, path):
    with open(path, "w", encoding = "utf-8") as file:
        for item in list:
            file.write(item + "\n")
 
########## Function to create movie folders. ##########
def folder_create(name):
    try:
        os.makedirs(folder_path, exist_ok = True)

    # If folder can not be created, terminate to check.
    except:
        print("Folder could not be created. Check again!")
        sys.exit()

########## Function to download the IMFDB images ##########
def download_images(images, name):
    
    # Set initial image count.
    count = 0
    #print(f"{len(images)} images found.")

    # Check if images are found.
    if len(images) != 0:
        # Skip first images, not downloading movie posters. 
        for i, image in enumerate(images[2:], start = 2):
            # Fetch image source from image tag.
            try:
                # Search for "data-srcset" in tag.
                image_link = image["data-srcset"]
            except:
                try:
                    # Search for "data-src" in tag. 
                    image_link = image["data-src"]
                except:
                    try:
                        # Search for "data-fallback-src" in tag.  
                        image_link = image["data-fallback-src"]
                    except:
                        try:
                            # Search for "src" in tag.   
                            image_link = image["src"]

                        # If no source URL is found, pass.
                        except:
                            pass          

            # Get image content after source URL is found.
            try:
                # Handle unneccesary images like logos.
                skip_keywords = skiplist
                skip = False

                # If image link contains skip keyword, image is not saved below.
                for keyword in skip_keywords:
                    if keyword.casefold() in image_link.casefold():
                        skip = True

                # Skip images with a width below 50px (all country flags + possible low size images).
                image_width = int(image.get("width", "0"))
                if image_width < config["width_threshold"]:
                    continue

                # Join image link with URL beginning to get full link.
                if (skip == False):
                    full_link = base_url + image_link
                    # Print if needed for debugging.
                    #print(full_link)
                    response = requests.get(full_link).content
                    try:
                        response = str(response, 'utf-8')
    
                    except UnicodeDecodeError:
                        # Start image download.
                        with open(f"{folder_path}/image{i+1}.jpg", "wb+") as file:
                            file.write(response)
                            
                        # Update image count.
                        count += 1
            except:
                pass

        print(f"{count}/{len(images)} images downloaded for movie {name}.")


#------------------------ EXTRACTION OF MOVIE NAMES ----------------------------

# Set script path as current dir and load imfdbConfig JSON file that stores parameters.
current_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
config_filepath = os.path.join(current_directory, "imfdbConfig.json")
with open(config_filepath, "r") as jsonfile:
    config = json.load(jsonfile)

# Define extraction URL from IMFDB website.
base_url = config["base_url"]
full_url = base_url + config["url_extension"] + config["url_category"]

# Creating movie name list starting at the first page.
print("Starting to extract movie name entries from IMFDb!")
all_names = extract_movie_names(full_url)

# As long as there is a next page, navigate to it, update URL
# and append new movie names to the created list.
while True:
    # Find the "next page" link.
    response = requests.get(full_url)
    soup = BeautifulSoup(response.content, "html.parser")
    next_page_element = soup.find("a", string="next page")

    # This part only executes if next page is present.
    if next_page_element:
        # Update the URL and extract new names.
        next_page_link = next_page_element.get("href")
        full_url = f"{base_url}{next_page_link}"
        current_names = extract_movie_names(full_url)

        # Append new names to the list.
        all_names.extend(current_names)

    # Break if there is no next page. 
    else:
        break

# Save movie names to textfile and print status.
movie_names_filepath = os.path.join(current_directory, config["movie_names_file"])
save_textfile(all_names, movie_names_filepath)
print("Saved " + str(len(all_names)) + " movie entries to \"" 
      + config["movie_names_file"] + "\"")


#------------------------ SCRAPING OF MOVIE IMAGES ----------------------------

# Define image destination and creating folder.
images_filepath = os.path.join(current_directory, config["image_folder"])
destination = images_filepath
os.makedirs(destination, exist_ok = True)

# Define keywoards that will later be skipped when scraping.
skiplist = config["skiplist"]

print("Starting to download movie images!")
# Iterate through movie names list.
for i, name in enumerate(all_names):

    # Define URL for each movie entry.
    movie_url = base_url + config["url_extension"] + name

    # Remove non-alphanumeric characters from name.
    folder_name = re.sub(r'[^\w\s]', '', name)
    # Define path and print status.
    folder_path = os.path.join(destination, folder_name)

    # Get and parse URL content, find all images.
    response = requests.get(movie_url)
    soup = BeautifulSoup(response.text, "html.parser")
    images = soup.findAll("img")

    # Create movie folder and download the images.
    folder_create(name)
    download_images(images, name)

    # Print download status every 10th iteration.
    if (i + 1) % 10 == 0:
        print(f"Download process: {i + 1}/{len(all_names)}")