"""
This script extracts arXiv identifiers from a Zotero SQLite database and writes them to a text file.

It first connects to the Zotero SQLite database located in the '~/Zotero' directory and fetches all rows from the 'itemDataValues' table. It then iterates over these rows and checks if any of them start with "https://arxiv.org/pdf" or "http://arxiv.org/pdf", indicating that they are links to arXiv PDFs. If a row is an arXiv link, the script extracts the arXiv identifier from the link and adds it to a set.

After all rows have been processed, the script changes the current working directory to '/home/mainuser/Desktop/LLMs/RagOverArXiv/data' and writes the arXiv identifiers to a text file named 'arxiv_names.txt', one identifier per line.

Finally, the script closes the connection to the Zotero SQLite database.
"""
import sqlite3
import os
os.chdir(os.path.expanduser('~/Zotero'))
conn = sqlite3.connect('zotero.sqlite')
cur = conn.cursor()
cur.execute('select * from itemDataValues')
rows = cur.fetchall()

arxiv_names = set()
for row_num,row in rows:
    if row.startswith("https://arxiv.org/pdf") or row.startswith("http://arxiv.org/pdf"):
        arxiv_names.add(row.split('/')[-1])
os.chdir(os.path.expanduser('/home/mainuser/Desktop/LLMs/RagOverArXiv/data'))
with open('arxiv_names.txt','w') as f:
    for name in arxiv_names:
        f.write(name+'\n')
    print(f"Wrote {len(arxiv_names)} arxiv names to arxiv_names.txt")
conn.close()