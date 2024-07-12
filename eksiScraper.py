import requests
from bs4 import BeautifulSoup
import csv
import time
from random import choice

headers_list = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
]

def scrape_eksisozluk(url, max_pages):
    entries = []
    for page in range(1, max_pages + 1):
        page_url = f'{url}?p={page}'
        success = False
        retries = 5
        while not success and retries > 0:
            try:
                headers = {'User-Agent': choice(headers_list)}
                response = requests.get(page_url, headers=headers, timeout=10)
                response.raise_for_status()
                success = True
            except (requests.exceptions.RequestException, requests.exceptions.ConnectTimeout) as e:
                retries -= 1
                print(f'Error scraping {page_url}, retries left: {retries}, error: {e}')
                time.sleep(5)  
        
        if not success:
            print(f'Failed to scrape {page_url} after several retries.')
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        
        
        print(f'Debug: HTML content of {page_url}')
        print(soup.prettify()[:1000])  

        page_entries = [entry.text.strip() for entry in soup.find_all(class_='content')]
        if not page_entries:
            break  
        entries.extend(page_entries)
        print(f'Scraped {len(page_entries)} entries from page {page} of {url}')
        
        time.sleep(1)  

    return entries


urls = {
    'https://eksisozluk.com/agorafobi--54730': 9,
    'https://eksisozluk.com/panik-atak--1337917': 155 
}

all_entries = []
for url, max_pages in urls.items():
    entries = scrape_eksisozluk(url, max_pages)
    

    title = url.split('--')[0].split('/')[-1].replace('-', ' ')
    for entry in entries:
        all_entries.append([title, entry])
    
    print(f'Scraped and collected {len(entries)} entries for {title}')


with open('eksisozluk_entries2.csv', 'w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Title', 'Entry'])
    writer.writerows(all_entries)

print('Scraping and saving to CSV complete.')
