#!/bin/bash

# python scripts/get_arxiv_nums.py 
# python scripts/write_arxiv_titles_and_abstracts.py
# python scripts/get_ak_articles.py --days_back 3
# Scrape: get paths to pass to below
start_time=$(date)
echo "Start time: $start_time"
# Find the latest file
latest_file=$(ls -t data/ak_articles_up_to/articles_up_to_*.csv | head -n1)

# Print the latest file
echo "Latest file: $latest_file"
script_dir=$(dirname "$0")
echo "Script dir: $script_dir"

python scripts/get_80-20_articles.py --directory_unread_csv $latest_file --directory_read_txts ./data/abstracts

# Loop over abstracts_new to get arXiv markup
# Move processed files to abstracts_new_processed or abstracts_new_cant_process
# Want to see progress at the command line, so doing this in bash instead of Python
for file in ./data/abstracts_new/*.txt; do
    filename=$(basename -- "$file" .txt)
    echo "Processing $filename"
    python ./scripts/get_arxiv_markup.py --arxiv_abbrev $filename --write_dir ./data/markups_new
    exit_status=$?
    if [ $exit_status -eq 0 ]; then
        echo "Success processing $filename. Moving to abstracts_new_processed."
        mv "$file" "./data/abstracts_new_processed/$filename.txt"
    else
        echo "Failed to process $filename, likely since Markup is not (yet?) available."
        mv "$file" "./data/abstracts_new_cant_process/$filename.txt"
    fi
done

python scripts/get_q_and_a.py --input_dir ./data/markups_new --output_dir ./data/q_and_a --processed_dir ./data/markups_new_processed --model_dir ../MixtralInference/Mixtral-8x7B-instruct-exl2


# # end_time=$(date)
# # echo "End time: $end_time"