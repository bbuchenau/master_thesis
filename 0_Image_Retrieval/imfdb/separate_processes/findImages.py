### This script crawls the IMFDB based on the extracted movie names and downloads all images. ###

from bs4 import *
import requests
import os
import inspect
import sys
import re

# Set working directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Set IMFDb URLs.
base_url = "https://www.imfdb.org/wiki/"
image_base = "https://www.imfdb.org"

# Items might be weapons (or the movies, if usecase 2).
items = []

# Load movie names textfile from extraction process.
with open("movieNames.txt", "r", encoding="utf-8") as file:
    for line in file:
        item = line.strip()
        items.append(item)

# Folder not uploaded to git, as it contains >100k images. Saved only locally.
folder_name = "imfdb_weapons"

# Images with names included in list are skipped. This is because IMFDb uses some
# images with the same name on each page, like a discord icon or logos.
skiplist = ["discord", "mediawiki", "logo", "poster", "spoilers"]

# Create folder where downloaded images are stored in.
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
weapons_folder = os.path.join(os.path.dirname(os.path.dirname(current_dir)), f"{folder_name}")
os.makedirs(weapons_folder, exist_ok=True)

# Iterate through weapons list.
for item in items:

    # Define full URL for each weapon.
    full_url = base_url + item

    # Function to create folder by weapon name.
    def folder_create(images):
        try:
            # Create the folder.
            folder_name = item

            # Remove non-alphanumeric characters from name.
            folder_name = re.sub(r'[^\w\s]', '', folder_name)

            folder_path = os.path.join(weapons_folder, folder_name)
            os.mkdir(folder_path)
            print(f"Folder '{folder_name}' created!")
            # Call function to download the images.
            download_images(images, folder_path)
    
        # If folder exists, exit.
        except:
            print("Folder name duplicate. Check again.")
            
    
    # Function to download the images.
    def download_images(images, folder_name):
    
        # Set initial image count.
        count = 0
    
        # Print total images found in URL
        print(f"{len(images)} images found.")
    
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
                    # TODO: Check optimal pixel threshold!
                    image_width = int(image.get("width", "0"))
                    if image_width < 50:
                        continue

                    # Join image link with URL beginning to get full link.
                    if (skip == False):
                        full_link = image_base + image_link
                        # Print if needed for debugging.
                        #print(image_link)
                        r = requests.get(full_link).content
                        try:
                            # Decoding to UTF-8.
                            r = str(r, 'utf-8')
        
                        except UnicodeDecodeError:
                            # Start image download.
                            with open(f"{folder_name}/image{i+1}.jpg", "wb+") as f:
                                f.write(r)
                                
                            # Update image count.
                            count += 1
                except:
                    pass
    
            # If all images are downloaded successfully.
            if count == len(images):
                print("All images downloaded.")
                
            # Else print share of downloaded images.
            else:
                print(f"{count} out of {len(images)} images downloaded.")
    
    # Main function to start crawling.
    def main(url):
    
        # Get URL content.
        r = requests.get(url)
    
        # Parse HTML code.
        soup = BeautifulSoup(r.text, 'html.parser')
    
        # Find all images.
        # TODO: Find out how to filter images by width/height attribute.
        images = soup.findAll('img')

        # Call folder create function
        folder_create(images)
    
    # Call main function.
    main(full_url)



