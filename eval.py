import os
import sys
import subprocess
import argparse

# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
parser.add_argument('--pdf_or_txt', type=str, required=True)
parser.add_argument('--chunk_size', type=int, required=True)
parser.add_argument('--chunk_overlap', type=int, required=True)
parser.add_argument('--nquestions', type=int, required=True)
parser.add_argument('--output_fn', type=str, required=True)
parser.add_argument('--embed_model_id', type=str, default='mixedbread-ai/mxbai-embed-large-v1')
parser.add_argument('--qagen_llm_dir', type=str, default="../MiStralInference", help='Path to the model directory')
parser.add_argument('--nquestions', type=int, default=10)
#parser.add_argument('--eval_output_path', type=str, default='./data/pdfs_ws_mrkp_test/eval_outputs/') #may not need to set since default points to right place
parser.add_argument('--eval_output_json', type=str, default='qa_couples.json') # --qas_dir!!!
parser.add_argument("--output_dir", type=str, default="./data/pdfs_ws_mrkp_test/eval_outputs/")
parser.add_argument('--critic_llm_dir', type=str, default="../MiStralInference", help='Path to the model directory')
args = parser.parse_args()

# Determine the files path and index path based on the pdf_or_txt argument
if args.pdf_or_txt == 'pdf':
    files_path = './data/pdfs_ws_mrkp_test/pdfs/'
    INDEX_PATH = './data/rag_index_dir/pdfs'
elif args.pdf_or_txt == 'txt':
    files_path = './data/pdfs_ws_mrkp_test/mrkps/'
    INDEX_PATH = './data/rag_index_dir/txts'
else:
    print("Invalid argument. Please specify 'pdf' or 'txt'.")
    sys.exit(1)

# Pieces for naming path
EMBED_MODEL = args.embed_model_id.split('/')[-1]
PDF_OR_TXT = args.pdf_or_txt
CHUNK_SIZE, CHUNK_OVERLAP = args.chunk_size, args.chunk_overlap
CRITIC_LLM = args.critic_llm_dir.split('/')[-1]
QAGEN_LLM = args.qagen_llm_dir.split('/')[-1]





# ----------------GET_VECTOR_STORE.PY--------------------
subprocess.run(['python', 'scripts/get_vector_store.py', '--pdf_or_txt=' + args.pdf_or_txt, '--files_path=' + files_path, '--chunk_size=' + args.chunk_size, '--chunk_overlap=' + args.chunk_overlap])

print("Generated vector store and stored at " + INDEX_PATH)

# ----------------GENERATE_EVAL_QA.PY-------------------- Generated JSON
subprocess.run(['python', 'scripts/generate_eval_qa.py', '--nquestions=' + args.nquestions, '--args.pdf_or_txt=' + args.pdf_or_txt, 
                '--output_file_name=' + args.output_fn, '--files_path=' + files_path, 
                '--chunk_size=' + args.chunk_size, '--chunk_overlap=' + args.chunk_overlap,
                '--qagen_llm_dir=' + args.qagen_llm_dir
                ])



parser.add_argument('--qagen_llm_dir', type=str, default="../MiStralInference", help='Path to the model directory')
parser.add_argument('--files_path', type=str, default='./data/pdfs_ws_mrkp_test/mrkps/')
parser.add_argument('--output_path', type=str, default='./data/pdfs_ws_mrkp_test/eval_outputs/')
parser.add_argument('--output_file_name', type=str, default='qa_couples.json')


QAS_PATH = files_path + args.output_fn

print("Index path in bash: " + INDEX_PATH)

# ----------------CRITIQUE_QA.PY-------------------- TODO
# parser.add_argument("--output_dir", type=str, default="./data/pdfs_ws_mrkp_test/eval_outputs/")
# parser.add_argument('--llm_dir', type=str, default="../MiStralInference", help='Path to the model directory')

#output_fn = 'MiStralInference_txt_critiqued_qas.csv'
parser.add_argument('--critic_output_file_name', type=str, default='critiqued_qas.csv')
# TODO: tidy up below!!!
critic_output_file_name = PDF_OR_TXT  + '_' + CRITIC_LLM + '_' + args.output_fn + '_critiqued_qas.csv'
subprocess.run(['python', 'scripts/critique_qa.py', '--qas_dir=' + args.eval_output_json, '--pdf_or_txt=' + args.pdf_or_txt,
                '--critic_llm_dir=' + args.critic_llm_dir, '--output_dir=' + args.output_dir,
                '--critic_output_file_name=' + critic_output_file_name])
                #'--output_file_name=' + output_fn, '--files_path=' + files_path])


parser.add_argument("--output_dir", type=str, default="./data/pdfs_ws_mrkp_test/eval_outputs/")
parser.add_argument('--critic_llm_dir', type=str, default="../MiStralInference", help='Path to the model directory')



# ----------------ANSWER_W_RAG_AND_TEST.PY-------------------- TODO
#RagOverArXiv/scripts/answer_w_rag_and_test.py
subprocess.run(['python', 'scripts/critique_qa.py', '--nquestions=' + nquestions, '--pdf_or_txt=' + pdf_or_txt, '--output_file_name=' + output_fn, '--files_path=' + files_path])

# ----------------JUDGE_ANSWERS.PY-------------------- TODO
#RagOverArXiv/scripts/judge_answers.py
subprocess.run(['python', 'scripts/judge_answers.py', '--nquestions=' + nquestions, '--pdf_or_txt=' + pdf_or_txt, '--output_file_name=' + output_fn, '--files_path=' + files_path])


# TODO: Make paths for outputs to be passed between subprocesses and modified after each step
# OR! Just make one final path once have df to add to!