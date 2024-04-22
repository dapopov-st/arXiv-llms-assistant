"""
scraper.py

This script uses Selenium to scrape AK's Daily Papers data from HuggingFace. 
It navigates through each paper link on a specific date's page, extracts the title, 
authors, and abstract of each paper, and stores this information in a pandas DataFrame.
See sample output in scrape/inspect_scraped_results.ipynb
Functions:
- get_one_days_df: Scrapes the data for one day and returns a DataFrame with the data.
- get_past_time_range_days_df: Initializes an empty DataFrame with the columns 'title', 'authors', and 'abstract'.

Setting Global Variables:

1. ENDING_DAY: This variable represents the date from which the script will start scraping data. By default, it is set to the current date. 
   If you want to start from a different date, comment out the current ENDING_DAY line and uncomment the next line, replacing "2023-11-01" 
   with your desired date in the format "YYYY-MM-DD".

2. DAYS_BACK: This variable represents the number of days back from the ENDING_DAY that the script will scrape data. You can adjust this 
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

import os
import re
import logging
from datetime import datetime, timedelta
import pandas as pd
import argparse
# Set up basic logging
logging.basicConfig(filename='./logs/scraper.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s')

# **********Get relevant dates to scrape**********
# Choose a starting day and how many days back to scrape
#ENDING_DAY = datetime.today().date() # Usual use case
# Adjust the day to start going back from different date, as in the commented out line below
#ENDING_DAY = datetime.strptime("2023-11-01","%Y-%m-%d").date() 
#DAYS_BACK = 8  # Can adjust number of days back to scrape
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

# **********Scraping AK's Articles from HuggingFace**********
# Next steps: 
# 1. Click through all the paper links on a specific date's page
# 2. Extract the abstract + title + authors of each
# 3. Store in a dataframe (if needed can write to a directory of text files, but this would be preferable)
# 4. Repeat for all dates in current_days

def get_one_days_df():
    """
    Scrapes the data for one day and returns a DataFrame with the data.

    This function navigates through each paper link on the current page, extracts the title, authors, and abstract of each paper,
    and stores this information in a pandas DataFrame. It continues this process until it encounters a TimeoutException,
    at which point it returns the DataFrame.

    Returns:
        DataFrame: A DataFrame with the columns 'title', 'authors', and 'abstract'. Each row represents one paper.
    """
    articles_df = pd.DataFrame(columns=['title','authors','abstract','arxiv_abbrev'])
    article_idx = 1
    while True:
        try:
            x_path = f'/html/body/div/main/div[2]/section/div[2]/div[{article_idx}]/article/div/div/div[2]/h3/a'
            element = WebDriverWait(browser, 10).until(
                EC.presence_of_element_located((By.XPATH, x_path)))

            element.click()
            page_text = browser.page_source
            match = re.search(r'https://arxiv\.org/pdf/(\d+\.\d+)', page_text)
            arxiv_abbrev = match.group(1) if match else None
            #print(f'Arxiv abbrev: {arxiv_abbrev}'); break
            title = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/main/div/section[1]/div/div[1]/h1')))
            authors = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/main/div/section[1]/div/div[1]/div[4]')))
            abstract = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.XPATH, '/html/body/div/main/div/section[1]/div/div[2]/p')))

            new_row = pd.DataFrame([{'title':title.text.replace('\n',' '), 'authors':authors.text.replace('\n',''),'abstract':abstract.text,'arxiv_abbrev':arxiv_abbrev}]) 
            articles_df = pd.concat([articles_df,new_row],ignore_index=True)
            browser.back()
            article_idx += 1

        except TimeoutException:
            #logging.exception('TimeoutException occurred')
            return articles_df
        except Exception as e:
            logging.exception('Exception occurred: \n',e)
        # finally:
        #     print(f"Arxiv abbrev: {arxiv_abbrev}"); break

def get_past_time_range_days_df(ending_day, days_back):
    """
    Initializes an empty DataFrame with the columns 'title', 'authors', and 'abstract'.

    This function is intended to be used as a starting point for scraping multiple days of data. The returned DataFrame can be
    used to store the data from multiple calls to get_one_days_df.

    Returns:
        None, writes a csv file to the data directory.
    """
    ARTICLES_DF = pd.DataFrame(columns=['title','authors','abstract'])
    # Get the past TIME_RANGE days excluding weekends; do reversed range so if TimeoutException occurs,
    # it occurs because there's no articles posted for starting day (yet)
    current_days = [datetime.strftime(ending_day-timedelta(days=i),"%Y-%m-%d") for i in reversed(range(0, days_back))
                    if (ending_day-timedelta(days=i)).weekday() not in [5,6]]
    for day in current_days:
        browser.get(f'https://huggingface.co/papers?date={day}')
        try:
            articles_df = get_one_days_df()
            ARTICLES_DF = pd.concat([ARTICLES_DF,articles_df],ignore_index=True)
            print(f'Finished scraping {day}')
        except TimeoutException: # In case articles have not been posted yet
            ARTICLES_DF.to_csv(f'./data/articles_up_to_{day}.csv',index=False)
            print('writing to csv')
    ARTICLES_DF.to_csv(f'./data/articles_up_to_{day}.csv',index=False)
    print('writing to csv')
    logging.info(f'Finished scraping all days for {ending_day} going {days_back} on {datetime.now()}')

parser = argparse.ArgumentParser(description="Get AK's article titles, authors, abstracts, and abbreviations for a given daterange (default today and 8 days back)")
parser.add_argument('--ending_day', type=str, default=datetime.today().date(),help='Ending date from which to start parsing backwards (default today)')
parser.add_argument('--days_back', type=str, default = 8,help='Number of days back to parse')
#parser.add_argument('--directory_unread_csv', type=str, default = './data/articles_up_to_2024-04-16.csv',help='Path to new articles csv file')
args = parser.parse_args() 

if __name__=='__main__':
    ending_day = args.ending_day if args.ending_day <= datetime.today().date() else datetime.today().date()
    days_back = int(args.days_back)
    get_past_time_range_days_df(ending_day, days_back)

    #TODO: FIX UP!!!