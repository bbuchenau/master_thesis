import snscrape.modules.twitter as sntwitter
import pandas as pd
import requests
import json
import os

# Set file path as directory.
os.chdir(os.path.dirname(__file__))

# Load config file that stores parameters.
with open("config.json", "r") as jsonfile:
    config = json.load(jsonfile)

# Set parameters.
search_tweet_limit = config["search_tweet_limit"]
media_tweet_limit = config["media_tweet_limit"]
date_query = "until:" + config["date_until_query"] + " since:" + config["date_since_query"]
geocode_query = "geocode:" + config["geocode_query"]
hashtags = config["hashtag_query"]

# Set folder for extracted list.
list_folder = "../../snscrape_lists/"
if not os.path.exists(list_folder):
        os.makedirs(list_folder)

# Loop through hashtags list for image extraction.
for hashtag in hashtags:

    # (Re)set tweets list storing data and id keeping track of tweet number.
    tweets = []
    id = 0

    # Set final query that is passed to sntwitter and storage folder.
    hashtag_query = "#" + hashtag
    image_folder = "../../snscrape_images/" + hashtag + "/"
    
    query = hashtag_query + " " + geocode_query
    print("TWITTER QUERY: " + query)

    # If image folder does not exist, create it, otherwise delete images it contains.
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    for f in os.listdir(image_folder):
        os.remove(os.path.join(image_folder, f))

    # Use TwitterSearchScraper to scrape data and append tweets to list.
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()): 
        if i > search_tweet_limit or len(tweets) > media_tweet_limit:
            break
        # Only extract data if media available.
        if (tweet.media): 
            tweets.append([hashtag, tweet.date, tweet.user.username, tweet.media])

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

        # Print progress (FIX!)
        if i == 1:
            print("Extracting " + str(media_tweet_limit) + " tweets with images for given query.")
            
    # Create dataframe from tweets list.
    colnames = ['Hashtag', 'Datetime', 'Username', 'Media']
    tweets_df = pd.DataFrame(tweets, columns = colnames)

    # Export to csv file.
    tweets_df.to_csv("../../snscrape_lists/" + hashtag + ".csv", mode = "w")