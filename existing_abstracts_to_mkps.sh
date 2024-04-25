#!/bin/bash

# Scrape: get paths to pass to below
start_time=$(date)
echo "Start time: $start_time"
# Find the latest file

# Loop over abstracts_new to get arXiv markup
# Move processed files to abstracts_new_processed or abstracts_new_cant_process
# Want to see progress at the command line, so doing this in bash instead of Python
for file in abstracts/*.txt; do
    filename=$(basename -- "$file" .txt)
    echo "Processing $filename"
    python ./scripts/get_arxiv_markup.py --arxiv_abbrev $filename --write_dir ./markups_existing
    exit_status=$?
    if [ $exit_status -eq 0 ]; then
        echo "Success processing $filename. Moving to abstracts_new_processed."
        mv "$file" "abstracts_new_processed/$filename.txt"
    else
        echo "Failed to process $filename, likely since Markup is not (yet?) available."
        mv "$file" "abstracts_new_cant_process/$filename.txt"
    fi
done


end_time=$(date)
echo "End time: $end_time"