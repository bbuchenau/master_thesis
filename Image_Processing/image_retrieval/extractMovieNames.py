import requests
import re
from bs4 import BeautifulSoup

# URL of the HTML website
url = "https://www.imfdb.org/wiki/Category:Movie"  # Replace with the actual URL

# Send a GET request to the website
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Find the <div> with class "mw-category"
div_tag = soup.find("div", id="mw-pages")

# Find all the <li> tags
li_tags = div_tag.find_all("li")

# Manipulate and store the modified strings in a list
list_elements = []
for tag in li_tags:
    text = tag.text
    # I can manipulate the movie name string here if wanted
    list_elements.append(text)

# Print the generated list
print(list_elements)