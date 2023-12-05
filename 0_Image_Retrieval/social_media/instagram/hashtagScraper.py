### This script uses the same logic as the geolocation instaScraper but for specific hashtags. ###

import snscrape.modules.instagram as sninstagram
from datetime import datetime
import inspect
import requests
import json
import os

# TODO: Create same for username to extract media from local news channels, if I need that.

# Main snscrape version not working for Instagram recently, GitHub
# pull request (fix #1001) working for one page only at a time.

########## Function to store post information into lists. ##########
def download_post_information(query, num_results):

      # Scrape posts according to location query and iterate.
      posts = sninstagram.InstagramHashtagScraper(query)
      for i, post in enumerate(posts.get_items()):
            print(post)
            if i >= num_results:
                  break

            # Set up dictionary for relevant information.
            post_dictionary = {
                  "id": post.id,
                  "date": post.date, 
                  "username" : post.username,
                  "content" : post.content,
                  "post_url" : post.url,
                  "is_video" : post.isVideo
            }

            # Store post information in metadata list.
            image_metadata.append(post_dictionary)

            # TODO: Fix part below, currently key error is raised because something in the delivered data is mixed up.

            # IMAGES: Handle if post has image.
            if not post.isVideo:
                  image_urls.append(post.medium.fullUrl)
                  post_dictionary["image_url"] = post.medium.fullUrl

            # VIDEOS: Handle if post has video.
            else:
                  image_urls.append(post.medium.variants[1].url)
                  post_dictionary["video_url"] = post.medium.variants[1].url

########## Function to save list to textfile. ##########
def save_textfile(list, path):
      with open(path, "a", encoding = "utf-8") as file:
                for item in list:
                      file.write(str(item) + "\n")

########## Function to extract and store post medium. ##########
def download_media(path, urls, metadata):
      for i, url in enumerate(urls):
            response = requests.get(url).content
            # Access post id for correct naming.
            current_id = metadata[i].get("id")
            try:
                response = str(response, 'utf-8')
            except UnicodeDecodeError:
                  # IMAGES: Save as image file.
                  if metadata[i].get("is_video") == False:
                        # Name is post id to keep track and prevent duplicates.
                        with open(f"{path}/{current_id}.jpg", "wb+") as file:
                              file.write(response)
                  # VIDEOS: Save as video file.
                  else:
                        # Name is post id to keep track and prevent duplicates.
                        with open(f"{path}/{current_id}.mp4", "wb+") as file:
                              file.write(response)

# --------------------- EXTRACTION OF INSTAGRAM IMAGES AND VIDEOS ----------------------

# Set script path as current dir and load location dictionary.
current_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
dictionary_filepath = os.path.join(current_directory, "location_dictionary.json")
with open(dictionary_filepath, "r") as jsonfile:
      location_dictionary = json.load(jsonfile)

# Store dictionary keys and values in lists.
hashtags = ["groupewagner"] # , "wagnergroup", "wagner", "ukrainewar", "sudanrevolts", "yemencrisis", "syriawar"

# FOR TESTING: Set maximum results.
num_results = 100

# Iterate through all locations and implement extraction process for posts.
for hashtag in hashtags:
    image_urls = []
    image_metadata = []
    
    # Start post information download.
    download_post_information(hashtag, num_results)

    # Save metadata as textfiles.
    url_path = os.path.join(current_directory, f"metadata_{hashtag}.txt")
    save_textfile(image_metadata, url_path)

    # Create folders and download media.
    images_folder = os.path.join(current_directory, f"{hashtag}")
    os.makedirs(images_folder, exist_ok = True)
    download_media(images_folder, image_urls, image_metadata)