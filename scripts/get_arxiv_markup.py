"""
scraper.py

This script uses Selenium to scrape AK's Daily Papers data from HuggingFace. 
It navigates through each paper link on a specific date's page, extracts the title, 
authors, and abstract of each paper, and stores this information in a pandas DataFrame.
See sample output in scrape/inspect_scraped_results.ipynb
Functions:
- get_one_days_df: Scrapes the data for one day and returns a DataFrame with the data.
- get_past_time_range_days_df: Initializes an empty DataFrame with the columns 'title', 'authors', and 'abstract'.

Script arguments:

1. ending_day: This variable represents the date from which the script will start scraping data. By default, it is set to the current date. 
   If you want to start from a different date, comment out the current ENDING_DAY line and uncomment the next line, replacing "2023-11-01" 
   with your desired date in the format "YYYY-MM-DD".

2. days_back: This variable represents the number of days back from the ENDING_DAY that the script will scrape data. You can adjust this 
   number as needed.

Configuring Selenium:

1. geckodriver_path: This variable should be set to the path of your geckodriver executable. Replace '/snap/bin/geckodriver' with the 
   actual path to your geckodriver executable.

2. The script then sets up a Firefox service with the specified geckodriver, creates a new Firefox options object, and creates a new 
   instance of the Firefox driver. You don't need to modify this code unless you want to use a different web driver or customize the 
   options.

DISCLAIMER:

This script is intended for educational purposes only. While the author did not find that scraping violates HuggingFace's terms of service, 
it is the user's responsibility to use this script in a manner that respects HuggingFace's terms of service and any applicable laws.

This script is designed to scrape a limited amount of data from arXiv at a time. Excessive use may lead to IP being 
blocked.

Use this script at your own risk. The author is not responsible for any consequences of using this script.
"""

# **********Importing Libraries**********
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import TimeoutException

from bs4 import BeautifulSoup

import os
import sys
import re
import logging
from datetime import datetime
import argparse
# Set up basic logging
logging.basicConfig(filename='./logs/get_arxiv_markup.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s')


os.makedirs('../data', exist_ok=True)
# **********Set up Selenium driver**********
# Specify the path to the geckodriver executable
geckodriver_path = '/snap/bin/geckodriver'  # replace with the actual path
# Create a Firefox service
s = Service(geckodriver_path)
# Create a new Firefox options object
options = Options()
# Create a new instance of the Firefox driver
browser = webdriver.Firefox(service=s, options=options)


def clean_text(page_text):
    page_text = page_text.replace(r'Report issue for preceding element','')
    cleaned_text = re.sub(r'\n\s*\n+', '\n', page_text)
    abstract_index=cleaned_text.upper().find('\nABSTRACT\n')
    ref_index=cleaned_text.upper().find('\nREFERENCES\n')
    if abstract_index==-1: abstract_index=0
    if ref_index==-1: ref_index=len(cleaned_text)
    return cleaned_text[abstract_index:ref_index]
    
def write_to_txt(cleaned_text,write_dir='./temp'):
    if cleaned_text:
       with open(f'{write_dir}/{arxiv_abbrev}_cleaned.txt','w') as f:
           f.write(cleaned_text) #page_text soup
           print(f'writing {arxiv_abbrev} to txt')
           logging.info(f'Finished writing {arxiv_abbrev} on {datetime.now()}')

def get_arxiv_markup(arxiv_abbrev):
    """
    Fetches the text content from the arXiv HTML page for a given paper.

    This function first attempts to fetch the v2 version of the paper. If the length of the text content is less than or equal to 1000 characters, it then tries to fetch the v1 version of the paper. If the length of the text content is still less than or equal to 1000 characters, it logs an error.

    Args:
        arxiv_abbrev (str): The abbreviation of the arXiv paper.

    Returns:
        str: The text content of the paper if it is longer than 1000 characters, otherwise None.

    Raises:
        TimeoutException: If a timeout occurs while trying to fetch the page.
        Exception: If any other exception occurs.
    """
    page_text=''
    try:
        # Try to navigate to the v2 URL
        browser.get(f'https://arxiv.org/html/{arxiv_abbrev}v2')
        page_source = browser.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        page_text = soup.get_text()
        # If the length of the text is less than 1000, navigate to the v1 URL
        if len(page_text) <= 2000:
            browser.get(f'https://arxiv.org/html/{arxiv_abbrev}v1')
            page_source = browser.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            page_text = soup.get_text()
            

    except TimeoutException:
        logging.exception('TimeoutException occurred')
        
    except Exception as e:
        logging.exception('Exception occurred: \n',e)
    finally:
        if len(page_text) <= 2000:
            logging.info(f'No text found for {arxiv_abbrev} on {datetime.now()}')
       
        return page_text if len(page_text) > 2000 else None


parser = argparse.ArgumentParser(description="Get article by arXiv number")
parser.add_argument('--arxiv_abbrev', type=str, help='Arxiv abbreviation number')
parser.add_argument('--write_dir', type=str, default='./temp', help='Directory for writing markup of arXiv papers')
args = parser.parse_args() 

if __name__=='__main__':
    arxiv_abbrev = args.arxiv_abbrev
    write_dir = args.write_dir
    if not os.path.exists(write_dir): os.makedirs(write_dir)

    cleaned_text,page_text = None,None
    page_text = get_arxiv_markup(arxiv_abbrev)
    if page_text: cleaned_text = clean_text(page_text)
    if cleaned_text:
        write_to_txt(cleaned_text,write_dir=write_dir)
        sys.exit(0) #success; can be used for further processing in bash script
    else:
        sys.exit(1) #failure
   
  
