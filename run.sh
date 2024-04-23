#!/bin/bash

# Scrape: get paths to pass to below
# Find the latest file
latest_file=$(ls -t data/articles_up_to_*.csv | head -n1)

# Print the latest file
echo "Latest file: $latest_file"
script_dir=$(dirname "$0")
echo "Script dir: $script_dir"

python scripts/get_80-20_articles.py --directory_unread_csv $latest_file --directory_read_txts ./abstracts

# Loop over abstracts_new to get arXiv markup
for file in abstracts_new/*.txt; do
    filename=$(basename -- "$file" .txt)
    echo "Processing $filename"
    python ./scripts/get_arxiv_markup.py --arxiv_abbrev $filename --write_dir ./markups_new
    exit_status=$?
    if [ $exit_status -eq 0 ]; then
        echo "Success"
        mv "$file" "abstracts_new_processed/$filename.txt"
    else
        echo "Failed to process $filename, likely since Markup is not (yet?) available."
        mv "$file" "abstracts_new_cant_process/$filename.txt"
    fi
done