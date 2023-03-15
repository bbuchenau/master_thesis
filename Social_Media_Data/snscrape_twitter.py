import snscrape.modules.twitter as sntwitter
import pandas as pd
import requests
import os

os.chdir(os.path.dirname(__file__))

# Set parameters.
query = "#Mali" # until:2020-01-01 since:2010-01-01
image_folder = "../../snscrape_images/"
search_limit = 10000
media_limit = 100

# Create list to append tweet data and initialize ids to track images.
tweets = []
id = 0

# If image folder does not exist, create it, otherwise delete images it contains.
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
for f in os.listdir(image_folder):
    os.remove(os.path.join(image_folder, f))

# Use TwitterSearchScraper to scrape data and append tweets to list.
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()): 
    if i > search_limit or len(tweets) > media_limit:
        break
    # Only extract data if media available.
    if (tweet.media): 
        tweets.append([tweet.date, tweet.user.username, tweet.media])

        # Save media in folder.
        for j, medium in enumerate(tweet.media):
            # Check if medium is photo and not video (for runtime and storage).
            if isinstance(medium, sntwitter.Photo):
                # Download image with get request.
                r = requests.get(medium.fullUrl)
                # Save as jpg named according to id and photo no within tweet.
                with open(image_folder + str(id) + "-" + str(j) + ".jpg", "wb") as fp:
                    fp.write(r.content)
        # Increase id             
        id = id + 1 

    if i % 50 == 0:
        print("Progress: " + str(len(tweets)) + " / " + str(media_limit) + " tweets with images extracted.")
        
    
# Create dataframe from tweets list.
tweets_df = pd.DataFrame(tweets, columns=['Datetime', 'Username', 'Media'])

# Export to csv file.
tweets_df.to_csv("snscrape_tweets.csv")