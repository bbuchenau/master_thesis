### This script crawls the IMFDB and saves all movie names into a textfile. ###

import os
import requests
from bs4 import BeautifulSoup

# Function to extract data from webpage.
def extract_data(url):
    # Send GET request.
    response = requests.get(url)

    # Parse HTML content using BeautifulSoup.
    soup = BeautifulSoup(response.content, "html.parser")

    # Find div with class "mw-category" and find all list tags.
    div_tag = soup.find("div", id="mw-pages")
    li_tags = div_tag.find_all("li")

    # Manipulate and store modified strings.
    list_elements = []
    for tag in li_tags:
        text = tag.text
        # Do I want to change movie name strings? If yes, here.
        list_elements.append(text)

    return list_elements

# Set working directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Set IMFDb URLs.
base_url = "https://www.imfdb.org"
url = "https://www.imfdb.org/wiki/Category:Movie"

# Extract data from first page.
all_data = extract_data(url)

# While there is a next page, extract also for for the next.
while True:
    # Find next page link.
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    next_page_elem = soup.find("a", string="next page")
    # If next page exists, continue.
    if next_page_elem:
        next_page_link = next_page_elem.get("href")
        # Update URL for the next iteration.
        url = f"{base_url}{next_page_link}"
        # Extract data from current page.
        current_data = extract_data(url)
        # Append new movie names to the list.
        all_data.extend(current_data)
    # Break if there is no next page.
    else:
        break

# Save movie name list as text file.
file_path = "movieNames.txt"
with open(file_path, "w", encoding="utf-8") as file:
    for movie_name in all_data:
        file.write(movie_name + "\n")

# Print list length (= number of movie names).
print(len(all_data))