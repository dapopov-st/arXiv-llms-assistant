import os
import sys
import subprocess

# # Get the command line arguments
# pdf_or_txt = sys.argv[1]
# CHUNK_SIZE = sys.argv[2]
# CHUNK_OVERLAP = sys.argv[3]
# NQUESTIONS = sys.argv[4]
# OUTPUT_FN = sys.argv[5]

# # Set the environment variables
# os.environ['CHUNK_SIZE'] = CHUNK_SIZE
# os.environ['CHUNK_OVERLAP'] = CHUNK_OVERLAP

import argparse

# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
parser.add_argument('--pdf_or_txt', type=str, required=True)
parser.add_argument('--chunk_size', type=int, required=True)
parser.add_argument('--chunk_overlap', type=int, required=True)
parser.add_argument('--nquestions', type=int, required=True)
parser.add_argument('--output_fn', type=str, required=True)

# Parse the arguments
args = parser.parse_args()

# Now you can use the arguments like this:
pdf_or_txt = args.pdf_or_txt
chunk_size = args.chunk_size
chunk_overlap = args.chunk_overlap
nquestions = args.nquestions
output_fn = args.output_fn



# Determine the files path and index path based on the pdf_or_txt argument
if pdf_or_txt == 'pdf':
    files_path = './data/pdfs_ws_mrkp_test/pdfs/'
    INDEX_PATH = './data/rag_index_dir/pdfs'
elif pdf_or_txt == 'txt':
    files_path = './data/pdfs_ws_mrkp_test/mrkps/'
    INDEX_PATH = './data/rag_index_dir/txts'
else:
    print("Invalid argument. Please specify 'pdf' or 'txt'.")
    sys.exit(1)

# Run the get_vector_store.py script
subprocess.run(['python', 'scripts/get_vector_store.py', '--pdf_or_txt=' + pdf_or_txt, '--files_path=' + files_path, '--chunk_size=' + chunk_size, '--chunk_overlap=' + chunk_overlap])

print("Generated vector store and stored at " + INDEX_PATH)

# Run the generate_eval_qa.py script
subprocess.run(['python', 'scripts/generate_eval_qa.py', '--nquestions=' + nquestions, '--pdf_or_txt=' + pdf_or_txt, '--output_file_name=' + output_fn, '--files_path=' + files_path])

QAS_PATH = files_path + output_fn

print("Index path in bash: " + INDEX_PATH)

# ----------------CRITIQUE_QA.PY--------------------
# parser.add_argument("--qas_dir", type=str, required=True)
# parser.add_argument('--pdf_or_txt', type=str, required=True)
# parser.add_argument("--output_dir", type=str, default="./data/pdfs_ws_mrkp_test/eval_outputs/")
# parser.add_argument('--llm_dir', type=str, default="../MiStralInference", help='Path to the model directory')

#output_fn = 'MiStralInference_txt_critiqued_qas.csv'
subprocess.run(['python', 'scripts/critique_qa.py', '--nquestions=' + nquestions, '--pdf_or_txt=' + pdf_or_txt, '--output_file_name=' + output_fn, '--files_path=' + files_path])
