"""
This script fetches the titles and abstracts of arXiv papers and writes them to text files.

It first reads the names of arXiv papers from a text file and the names of existing abstract files from a directory. It then subtracts the set of existing files from the set of arXiv paper names to get the set of papers for which it needs to fetch the abstracts.

For each paper in this set, it constructs a URL for the arXiv API and fetches the data. It then extracts the title and abstract from the data and writes them to a text file in the 'abstracts' directory. The name of the text file is the arXiv ID of the paper.

The `get_title_and_abstract` function is a helper function that fetches the title and abstract of a paper given its arXiv ID. It returns the title and abstract as a tuple.

This script should be run as a standalone script.
"""

import feedparser
import os
from utils import get_title_and_abstract
# Get titles
existing_files = set(os.listdir('./data/abstracts'))
with open('./data/arxiv_names.txt','r') as f:
    arxiv_ids = set(".".join(line.strip().split('.')[:2])+'.txt' for line in f) 

# Subtract out the existing files
arxiv_ids = arxiv_ids - existing_files
#print(arxiv_ids)
#print(f"Len of arxiv ids: {len(arxiv_ids)}")


if __name__ == '__main__':
    for arxiv_id in arxiv_ids:
        arxiv_id = arxiv_id.replace('.txt','').replace('.pdf','')
        title, abstract = get_title_and_abstract(arxiv_id)
        if title and abstract:
            with open(f'./data/abstracts/{arxiv_id}.txt','w') as f:
                f.write(title+'\n'+abstract)
            print(f"Wrote {arxiv_id} to file.")
            


