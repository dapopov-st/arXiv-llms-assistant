#!/bin/bash

# Scrape: get paths to pass to below
# Find the latest file
latest_file=$(ls -t data/articles_up_to_*.csv | head -n1)

# Print the latest file
echo "Latest file: $latest_file"
script_dir=$(dirname "$0")
echo "Script dir: $script_dir"
#cd ./scripts
# Make embeds
python scripts/get_80-20_articles.py --directory_unread_csv $latest_file --directory_read_txts ./abstracts
# Find top K
