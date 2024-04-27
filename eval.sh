# Break up Python scripts into smaller scripts returning outputs to be used in the main script
# The purpose is to free up GPU resources between steps

pdf_or_txt=$1
CHUNK_SIZE=$2
CHUNK_OVERLAP=$3
NQUESTIONS=$4
OUTPUT_FN=$5
#./eval.sh pdf 2000 200
# Export the variables as environment variables
export CHUNK_SIZE
export CHUNK_OVERLAP


# if [ "$pdf_or_txt" = "pdf" ]; then
#     python scripts/get_vector_store.py --pdf_or_txt='pdf' --files_path='./data/pdfs_ws_mrkp_test/pdfs/' --chunk_size=$CHUNK_SIZE --chunk_overlap=$CHUNK_OVERLAP
#     INDEX_PATH='./data/rag_index_dir/pdfs'
# elif [ "$pdf_or_txt" = "txt" ]; then
#     python scripts/get_vector_store.py --pdf_or_txt='txt' --files_path='./data/pdfs_ws_mrkp_test/mrkps/' --chunk_size=$CHUNK_SIZE --chunk_overlap=$CHUNK_OVERLAP
#     INDEX_PATH='./data/rag_index_dir/txts'
# else
#     echo "Invalid argument. Please specify 'pdf' or 'txt'."
#     exit 1
# fi
# echo "Generated vector store and stored at ${INDEX_PATH}"

if [ "$pdf_or_txt" = "pdf" ]; then
    python scripts/generate_eval_qa.py --nquestions=$NQUESTIONS --pdf_or_txt='pdf' --output_file_name=$OUTPUT_FN --files_path='./data/pdfs_ws_mrkp_test/pdfs/'
    files_path='./data/pdfs_ws_mrkp_test/mrkps/'
    QAS_PATH="${files_path}${OUTPUT_FN}"
elif [ "$pdf_or_txt" = "txt" ]; then
    python scripts/generate_eval_qa.py --nquestions=$NQUESTIONS --pdf_or_txt='txt' --output_file_name=$OUTPUT_FN --files_path='./data/pdfs_ws_mrkp_test/mrkps/'
    files_path='./data/pdfs_ws_mrkp_test/mrkps/'
    QAS_PATH="${files_path}${OUTPUT_FN}"
else
    echo "Invalid argument. Please specify 'pdf' or 'txt'."
    exit 1
fi


echo "Index path in bash: ${INDEX_PATH}"