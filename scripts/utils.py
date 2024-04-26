import re
import numpy as np
import logging
import feedparser
import os
import shutil
def classify_topic_re(example):
    """Classify abstract based on regex pattern"""
    pattern = r'^(?!.*\b(diffusion|3D|computer vision|image|video|resnet|cnn|vit)\b).*\b(lms|attention|language model(s)?|LLMs|context LMs|synthetic data|GPT|RLHF|DPO|KTO|ORPO|.*RNN.*|llama|mamba).*'
    match = re.findall(pattern, example['abstract'], re.IGNORECASE)
    return 'llms' if match else 'other'

def get_title(arxiv_id):
    """
    Fetches the title of a paper from the arXiv API.

    This function constructs a query URL using the provided arXiv ID, fetches the data from the arXiv API, and extracts the title of the paper. If an error occurs, it logs the error and returns an empty string.

    Args:
        arxiv_id (str): The arXiv ID of the paper.

    Returns:
        str: The title of the paper, or an empty string if an error occurs.
    """
    logging.basicConfig(filename='../logs/get_title.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s')
    try:
        url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'

        # Fetch the data
        data = feedparser.parse(url)

        # Get the first entry
        entry = data.entries[0]

        # Get title and abstract
        title = entry.title.replace('\n', '').replace('\r', '').strip()
    
        return title
    except Exception as e:
        print(f"Error: {e}")
        logging.exception(f'Error occurred in getting title for {arxiv_id}: {e}')
        return ''
    
def get_title_and_abstract(arxiv_id):
    # Construct the query URL
    try:
        url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'

        # Fetch the data
        data = feedparser.parse(url)

        # Get the first entry
        entry = data.entries[0]

        # Get title and abstract
        title = entry.title.replace('\n', '').replace('\r', '').strip()
        abstract = entry.summary
        return title, abstract
    except Exception as e:
        print(f"Error: {e}")
        return '',''
    
def move_processed_files(input_dir, processed_dir, filename):
    """
    Moves a processed file from one directory to another.

    This function constructs the full paths to the input and output files and then moves the file from the input directory to the processed directory.

    Args:
        input_dir (str): The directory where the input file is currently located.
        processed_dir (str): The directory where the processed file should be moved to.
        filename (str): The name of the file to be moved.

    Returns:
        None
    """
    input_file = os.path.join(input_dir, filename)
    processed_file = os.path.join(processed_dir, filename)
    shutil.move(input_file, processed_file)