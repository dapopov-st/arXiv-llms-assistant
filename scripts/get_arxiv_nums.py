"""
This script extracts arXiv identifiers from a Zotero SQLite database and writes them to a text file.

It first connects to the Zotero SQLite database located in the '~/Zotero' directory and fetches all rows from the 'itemDataValues' table. It then iterates over these rows and checks if any of them start with "https://arxiv.org/pdf" or "http://arxiv.org/pdf", indicating that they are links to arXiv PDFs. If a row is an arXiv link, the script extracts the arXiv identifier from the link and adds it to a set.

After all rows have been processed, the script changes the current working directory to '/home/mainuser/Desktop/LLMs/RagOverArXiv/data' and writes the arXiv identifiers to a text file named 'arxiv_names.txt', one identifier per line.

Finally, the script closes the connection to the Zotero SQLite database.
"""
import sqlite3
import os
import argparse

def get_arxiv_nums(zotero_path):
    original_dir = os.getcwd()
    os.chdir(os.path.expanduser(zotero_path))

    conn = sqlite3.connect('zotero.sqlite')
    cur = conn.cursor()
    try:
        cur.execute('select * from itemDataValues')
    except sqlite3.OperationalError as e:
        if 'database is locked' in str(e):
            print('Database is locked, please close Zotero and retry...')
    rows = cur.fetchall()

    arxiv_names = set()
    for _,row in rows:
        if row.startswith("https://arxiv.org/pdf") or row.startswith("http://arxiv.org/pdf"):
            arxiv_names.add(row.split('/')[-1])

    os.chdir(original_dir)
    os.chdir(os.path.expanduser('./data'))
    
    with open('arxiv_names.txt','w') as f:
        for name in arxiv_names:
            f.write(name+'\n')
        print(f"Wrote {len(arxiv_names)} arxiv names to arxiv_names.txt")
    conn.close()


parser = argparse.ArgumentParser(description='Get arxiv numbers from Zotero')
parser.add_argument('--zotero_path', type=str, default='~/Zotero', help='Path to Zotero files on your system')
args = parser.parse_args()
if __name__ == '__main__':
    zotero_path = args.zotero_path
    get_arxiv_nums(zotero_path)

