import requests
from bs4 import BeautifulSoup
import csv
import time
from random import choice

# List of user-agent strings to simulate different browsers for requests
headers_list = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
]

def scrape_eksisozluk(url, max_pages):
    entries = []  # Initialize a list to store the entries
    for page in range(1, max_pages + 1):
        page_url = f'{url}?p={page}'  # Construct the URL for each page
        success = False
        retries = 5  # Number of retries for failed requests
        while not success and retries > 0:
            try:
                # Randomly select a user-agent string from the list
                headers = {'User-Agent': choice(headers_list)}
                # Send an HTTP GET request to the page URL
                response = requests.get(page_url, headers=headers, timeout=10)
                # Raise an exception if the request was unsuccessful
                response.raise_for_status()
                success = True  # Mark as successful if no exceptions were raised
            except (requests.exceptions.RequestException, requests.exceptions.ConnectTimeout) as e:
                # Decrement retries count and print error message if request fails
                retries -= 1
                print(f'Error scraping {page_url}, retries left: {retries}, error: {e}')
                time.sleep(5)  # Wait for 5 seconds before retrying
        
        if not success:
            # If all retries are exhausted, print a failure message and exit the loop
            print(f'Failed to scrape {page_url} after several retries.')
            break

        # Parse the page content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Debugging print statement to show the HTML content of the page (truncated to 1000 characters)
        print(f'Debug: HTML content of {page_url}')
        print(soup.prettify()[:1000])  

        # Extract text from all elements with the class 'content' and strip whitespace
        page_entries = [entry.text.strip() for entry in soup.find_all(class_='content')]
        if not page_entries:
            # If no entries are found, exit the loop
            break  
        # Add the entries from this page to the list of all entries
        entries.extend(page_entries)
        print(f'Scraped {len(page_entries)} entries from page {page} of {url}')
        
        # Wait for 1 second between page requests to avoid overwhelming the server
        time.sleep(1)  

    return entries

# Dictionary of URLs to scrape with the maximum number of pages for each
urls = {
    'https://eksisozluk.com/distimi--306507': 6,
    'https://eksisozluk.com/distimik-bozukluk--1155620': 2,
    'https://eksisozluk.com/depresyon--33371': 400,
    'https://eksisozluk.com/depresyon-belirtileri--248540': 23 
}

all_entries = []  # Initialize a list to store all entries from all URLs
for url, max_pages in urls.items():
    entries = scrape_eksisozluk(url, max_pages)  # Scrape entries for each URL
    
    # Extract the title from the URL
    title = url.split('--')[0].split('/')[-1].replace('-', ' ')
    # Add each entry with its title to the list of all entries
    for entry in entries:
        all_entries.append([title, entry])
    
    print(f'Scraped and collected {len(entries)} entries for {title}')

# Write all collected entries to a CSV file
with open('eksisozluk_entries1.csv', 'w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Title', 'Entry'])  # Write header row
    writer.writerows(all_entries)  # Write all entries

print('Scraping and saving to CSV complete.')
