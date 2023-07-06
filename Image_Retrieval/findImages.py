# This script crawls the IMFDB and downloads the images set in the items list 
# into a set folder. Base and image URLs can be adjusted for different use cases.


from bs4 import *
import requests
import os
import inspect
import sys
import re

# Set URL base, image URL base, item list and folder name.
base_url = "https://www.imfdb.org/wiki/"
image_base = "https://www.imfdb.org"

# Items might be weapons (tested) or the movies (to be tested)
items = ['"Cyclone" Will Begin at Night ("Tsiklon" nachnyotsya nochyu)', "'71", "'Burbs, The", '008: Operation Exterminate', '009-1: The End of the Beginning', '10 Cloverfield Lane', '10 Minutes Gone', '10 to Midnight', '100 Bloody Acres', '100 Days Before the Command (Sto dney do prikaza...)', '100 Rifles', '10th Victim, The', '11.6', 
'12 Rounds', '12 Rounds 2: Reloaded', '12 Rounds 3: Lockdown', '12 Strong', '12 Years a Slave', '13', '13 Hours: The Secret Soldiers of Benghazi', '13 Minutes', '1492: Conquest of Paradise', '15 Minutes', "15 Minutes of War (L'Intervention)", '15:17 to Paris, The', '16 Blocks', '1612', '18-14', '1911 (2011)', '1917 (2019)', '1922 (2017)', '1941', '1944', '1968 Tunnel Rats', '2 Days in the Valley', '2 Fast 2 Furious', '2 Guns', '2 Lava 2 Lantula!', '20 Million Miles to Earth', '20000 Leagues Under the Sea', '2009: Lost Memories', '2012', '21 Bridges', '21 Grams', '21 Jump Street (2012)', '22 Bullets', '22 July', '22 Jump Street', '22 Minutes (22 minuty)', '23 (1998)', '23-F', '24 Hours (24 chasa)', '24-25 Does Not Return (24-25 ne vozvrashchaetsya)', '24: Redemption', '25th Hour', '28 Days Later', '28 Weeks Later', '3 Days to Kill', '3 From Hell', '3 Women', '30 Days of Night', '30 Minutes or Less (2011)', '3000 Miles to Graceland', '31 North 62 East', '317th Platoon, The', '355, The', '36th Precinct', '39 Steps, The (1935)', '39 Steps, The (1959)', '39 Steps, The (2008)', '3:10 to Yuma (1957)', '3:10 to Yuma (2007)', '4 for Texas', '4.3.2.1', '44 Minutes: The North Hollywood Shootout', '47 Ronin', '48 Hrs.', '5 Days of War', '52 Pick-Up', '55 Days at Peking', '5th Wave, The', '6 Days', '6 Guns', '6 Underground', '633 Squadron', '6th Day, The', '7 Days in Entebbe', '7 Seconds', '7 Witches', '71: Into the Fire', '7th Heaven', '8 Heads in a Duffel Bag', '8 Mile', 
'8 Million Ways to Die', '800 Bullets (800 Malas)', '88 Minutes', '8MM', '99 and 44/100% Dead', '999', '9th Company', 'A Bad Good Man (Plokhoy khoroshiy chelovek)', 
'A Bay of Blood', 'A Beautiful Mind', 'A Better Tomorrow', 'A Better Tomorrow II', 'A Better Tomorrow III', 'A Bittersweet Life', 'A Boy and His Dog', 'A Breath of Scandal', 'A Bridge Too Far', 'A Bronx Tale', 'A Bullet For Joey', 'A Bullet for Pretty Boy', 'A Bullet for the General', 'A Call to Spy', 'A Captain at Fifteen (Pyatnadtsatiletniy kapitan)', "A Captain's Honor (L'Honneur d'un capitaine)", 'A Christmas Story', 'A Clockwork Orange', 'A Colt Is My Passport', 'A Conspiracy of Faith', 'A Cruel Romance', 'A Dandy in Aspic', 'A Dangerous Man', 'A Dark Truth', 'A Day of Fury', 'A Dear Boy (Dorogoy malchik)', 'A Farewell to Arms (1932)', 'A Farewell to Arms (1957)', 'A Few Days in September', 'A Few Good Men', 'A Field in England', 'A Fish Called Wanda', 'A Fistful of Dollars', 'A Force of One', 'A Game without Rules (Hra bez pravidel)', 'A Gang Story (Les Lyonnais)', 'A Generation (Pokolenie)', 'A Gentle Creature (Krotkaya)', 'A Gentle Woman (Une femme douce)', 'A Golden-coloured Straw Hat (Solomennaya shlyapka)', 'A Good Day to Die Hard', 'A Good Lad (Slavnyy malyy)', 'A Good Man', 'A Hard Day (2014)', 'A Hard Day (2021)', 'A Hill in Korea', 'A History of Violence', 'A Hologram for the King', 'A Holy Place (Sveto mesto)', 'A Janitor', "A Jester's Tale (Bláznova kronika)", 'A Journal for Jordan', 'A Judgement in Stone (La Cérémonie)', 'A Life Less Ordinary', 'A Low Down Dirty Shame', 'A Man Apart', 'A Man Before His Time (Prezhdevremennyy chelovek)', 'A Man Called Blade (Mannaja)', 'A Man Called Magnum (Napoli si ribella)', 'A Man from the Boulevard des Capucines (Chelovek s bulvara Kaputsinov)', 'A Man Named Rocca (Un nommé La Rocca)', 'A Midnight Clear', 'A Million Ways to Die in the West', 'A Most Violent Year', 'A Night to Remember', 'A Nightmare on Elm Street (1984)', "A Nightmare on Elm Street 2: Freddy's Revenge", 'A Noisy Household (Bespokoynoe khozyaystvo)', "A Pain in the Ass (L'emmerdeur) (1973)", "A Pain in the Ass (L'emmerdeur) (2008)", 'A Perfect Getaway', 'A Perfect Murder', 'A Perfect World', 'A Pistol Shot (Vystrel)', 'A Police Commissioner Accuses (Un comisar acuza)', 'A Prayer for Katarina Horovitzova', 'A Professional Gun (Il mercenario)', 'A Prophet', 'A Quiet Outpost (Tikhaya zastava)', 'A Royal Night Out', 'A Scanner Darkly', 'A Serbian Film', 'A Shot in the Dark', 'A Simple Plan', 'A Slight Case of Murder', "A Soldier's Story", 'A Sound of Thunder', 'A Special Cop in Action (Italia a mano armata)', 'A Star Called Wormwood (Hvězda zvaná Pelyněk)', 'A Step into the Darkness (Krok do tmy)', 'A Stranger Among Us', 'A Study in Terror', 'A Tale About Nipper-Pipper (Skazka o Malchishe-Kibalchishe)', 'A Time to Kill', 'A Time to Love and a Time to Die', 'A Trap for Jackals (Kapkan dlya shakalov)', 'A Twelve-Year Night', 'A Very Harold & Kumar 3D Christmas', 'A Very Long Engagement']
folder_name = "imfdb_images_thumb_tnone"

# Images with names included in list are skipped.
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
    
        # If folder exists, exit.
        except:
            print("Folder name duplicate. Check again.")
            sys.exit()
    
        # Call function to download the images.
        download_images(images, folder_path)
    
    
    # Function to download the images.
    def download_images(images, folder_name):
    
        # Set initial image count.
        count = 0
    
        # Print total images found in URL
        print(f"{len(images)} images found.")
    
        # Check if images are found.
        if len(images) != 0:
            for i, image in enumerate(images):
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



