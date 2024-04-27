# Break up Python scripts into smaller scripts returning outputs to be used in the main script
# The purpose is to free up GPU resources between steps

pdf_or_txt=$1

if [ "$pdf_or_txt" = "pdf" ]; then
    python scripts/get_vector_store.py --pdf_or_txt='pdf' --files_path='./data/pdfs_ws_mrkp_test/pdfs/'
    INDEX_PATH='./data/rag_index_dir/pdfs'
elif [ "$pdf_or_txt" = "txt" ]; then
    python scripts/get_vector_store.py --pdf_or_txt='txt' --files_path='./data/pdfs_ws_mrkp_test/mrkps/'
    INDEX_PATH='./data/rag_index_dir/txts'
else
    echo "Invalid argument. Please specify 'pdf' or 'txt'."
    exit 1
fi

echo "Index path in bash: ${INDEX_PATH}"