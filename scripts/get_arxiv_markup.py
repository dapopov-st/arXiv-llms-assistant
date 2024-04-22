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

This script is designed to scrape a limited amount of data (e.g., a week's worth) at a time. Excessive use may lead to your IP being 
blocked by HuggingFace.

Furthermore, HuggingFace may present "I'm not a robot" challenges to confirm that the user is a human. The user should be present when 
the script is running to respond to these challenges.

Use this script at your own risk. The author is not responsible for any consequences of using this script.
"""

# **********Importing Libraries**********
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from bs4 import BeautifulSoup

import os
import re
import logging
from datetime import datetime, timedelta
import pandas as pd
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
    page_text=page_text.replace(r'Report issue for preceding element','')
    cleaned_text = re.sub(r'\n\s*\n+', '\n', page_text)
    abstract_index=cleaned_text.upper().find('\nABSTRACT\n')
    ref_index=cleaned_text.upper().find('\nREFERENCES\n')
    cleaned_text= cleaned_text[abstract_index:ref_index]
    return cleaned_text
    
def write_to_txt(clean_text):
    if clean_text:
       with open(f'./temp/{arxiv_abbrev}_cleaned.txt','w') as f:
           f.write(clean_text) #page_text soup
           print(f'writing {arxiv_abbrev} to txt')
           logging.info(f'Finished writing {arxiv_abbrev} on {datetime.now()}')
    else:
        logging.error(f'No text found for {arxiv_abbrev} on {datetime.now()}')
        
def get_arxiv_markup(arxiv_abbrev):
    """
    Scrapes the data for one day and returns a DataFrame with the data.

    This function navigates through each paper link on the current page, extracts the title, authors, and abstract of each paper,
    and stores this information in a pandas DataFrame. It continues this process until it encounters a TimeoutException,
    at which point it returns the DataFrame.

    Returns:
        DataFrame: A DataFrame with the columns 'title', 'authors', and 'abstract'. Each row represents one paper.
    """
   
    
    try:
        browser.get(f'https://arxiv.org/html/{arxiv_abbrev}v1')
        

        page_source = browser.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        # Extract the text
        page_text = soup.get_text()
        references_index= page_text.upper().find('\nREFERENCES\n')
        page_text = page_text[:references_index]
  
        return page_text if page_text else None
       

    except TimeoutException:
        #logging.exception('TimeoutException occurred')
        return
    except Exception as e:
        logging.exception('Exception occurred: \n',e)



parser = argparse.ArgumentParser(description="Get article by arXiv number")
parser.add_argument('--arxiv_abbrev', type=str, help='Arxiv abbreviation number')
#parser.add_argument('--directory_unread_csv', type=str, default = './data/articles_up_to_2024-04-16.csv',help='Path to new articles csv file')
args = parser.parse_args() 

if __name__=='__main__':
    arxiv_abbrev = args.arxiv_abbrev
  
    clean_text = clean_text(get_arxiv_markup(arxiv_abbrev))
    if clean_text: write_to_txt(clean_text)
   
    #TODO: Make sure to check post Dec 2023; check if get legitimate response, not 'HTML is not available for the source.'
    #TODO: check v2 (think seen it somewhere)
