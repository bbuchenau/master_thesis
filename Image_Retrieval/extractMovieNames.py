import requests
import re
from bs4 import BeautifulSoup




# Function to extract data from the current page
def extract_data(url):
    # Send a GET request to the current page
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup.
    # Creating soup initializes the extraction process.
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the <div> with class "mw-category"
    div_tag = soup.find("div", id="mw-pages")

    # Find all the <li> tags
    li_tags = div_tag.find_all("li")

    # Manipulate and store the modified strings in a list
    list_elements = []
    for tag in li_tags:
        text = tag.text
        # You can manipulate the movie name string here if needed
        list_elements.append(text)

    return list_elements

# URL of the HTML website
base_url = "https://www.imfdb.org"
url = "https://www.imfdb.org/wiki/Category:Movie"  # Replace with the actual URL

# Extract data from the first page
all_data = extract_data(url)

# Loop to extract data from all pages
while True:
    # Find the "next page" link
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    next_page_elem = soup.find("a", string="next page")
    # This part only executes if next page is present.
    if next_page_elem:
        next_page_link = next_page_elem.get("href")
        # Update the URL for the next iteration
        url = f"{base_url}{next_page_link}"
        # Extract data from the current page
        current_data = extract_data(url)
        # Append the current page data to the list
        all_data.extend(current_data)
    # Break if crawler arrived at last page. 
    else:
        break

# Save output list as text file.
file_path = "movieNames.txt"
with open(file_path, "w", encoding="utf-8") as file:
    for movie_name in all_data:
        file.write(movie_name + "\n")

# Print the list containing data from all pages
print(len(all_data))