import os
import sys
import subprocess
import json
import argparse

# Load the configuration file
with open('config.json', 'r') as f:
    config = json.load(f)


# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
parser.add_argument('--pdf_or_txt', type=str, default=config['pdf_or_txt'])
parser.add_argument('--chunk_size', type=int, default=config['chunk_size'])
parser.add_argument('--chunk_overlap', type=int, default=config['chunk_overlap'])
parser.add_argument('--nquestions', type=int, default=config['nquestions'])
parser.add_argument('--embed_model_id', type=str, default=config['embed_model_id'])
parser.add_argument('--qagen_llm_dir', type=str, default=config['qagen_llm_dir'])
parser.add_argument('--eval_output_dir', type=str, default=config['eval_output_dir'])
#parser.add_argument('--eval_output_json', type=str, default=config['eval_output_json']) # --qas_dir!!!
parser.add_argument('--critic_llm_dir', type=str, default=config['critic_llm_dir'])
#parser.add_argument('--critic_output_file_name', type=str, default='critiqued_qas.csv')
parser.add_argument('--reader_llm_dir', type=str, default=config['reader_llm_dir'], help='Path to the model directory')

#parser.add_argument("--output_dir", type=str, default="./data/pdfs_ws_mrkp_test/eval_outputs/")
#parser.add_argument('--critic_llm_dir', type=str, default="../MiStralInference", help='Path to the model directory')
args = parser.parse_args()

# Determine the files path and index path based on the pdf_or_txt argument
if args.pdf_or_txt == 'pdf':
    FILES_PATH = './data/pdfs_ws_mrkp_test/pdfs/'
    INDEX_PATH = './data/rag_index_dir/pdfs'
elif args.pdf_or_txt == 'txt':
    FILES_PATH = './data/pdfs_ws_mrkp_test/mrkps/'
    INDEX_PATH = './data/rag_index_dir/txts'
else:
    print("Invalid argument. Please specify 'pdf' or 'txt'.")
    sys.exit(1)

# Pieces for naming path
EMBED_MODEL = args.embed_model_id.split('/')[-1][:10]
PDF_OR_TXT = args.pdf_or_txt
CHUNK_SIZE, CHUNK_OVERLAP = str(args.chunk_size), str(args.chunk_overlap)
CRITIC_LLM = 'Critic'+args.critic_llm_dir.split('/')[-1][:10]
QAGEN_LLM = 'Qagen'+args.qagen_llm_dir.split('/')[-1][:10]




QAS_NAME = "_".join([PDF_OR_TXT,CHUNK_SIZE, CHUNK_OVERLAP ,EMBED_MODEL,CRITIC_LLM,QAGEN_LLM])

QAS_NAME_JSON, QAS_NAME_DF = QAS_NAME+'.json',QAS_NAME+'.csv'
QAS_PATH_JSON, QAS_PATH_DF = os.path.join(args.eval_output_dir,QAS_NAME_JSON),os.path.join(args.eval_output_dir,QAS_NAME_DF)
# print("Index path in bash: " + INDEX_PATH)
# print("Qas path in bash: " + QAS_NAME_JSON,QAS_NAME_DF)
# print("Qas path in bash: " + QAS_PATH_JSON,QAS_PATH_DF)



# ----------------GET_VECTOR_STORE.PY--------------------
subprocess.run(['python', 'scripts/get_vector_store.py', '--pdf_or_txt=' + args.pdf_or_txt, '--files_path=' + FILES_PATH, 
                '--chunk_size=' + str(args.chunk_size), '--chunk_overlap=' + str(args.chunk_overlap)])

print("Generated vector store and stored at " + INDEX_PATH)

# ----------------GENERATE_EVAL_QA.PY-------------------- Generated JSON
subprocess.run(['python', 'scripts/generate_eval_qa.py', '--nquestions=' + str(args.nquestions), '--pdf_or_txt=' + args.pdf_or_txt, 
               '--chunk_size=' + str(args.chunk_size), '--chunk_overlap=' + str(args.chunk_overlap), 
                '--input_files_dir=' + FILES_PATH, #'--eval_output_path=' + args.eval_output_path,
                '--qagen_llm_dir=' + args.qagen_llm_dir,  '--eval_output_fullpath=' + QAS_PATH_JSON])







# ----------------CRITIQUE_QA.PY-------------------- TODO
subprocess.run(['python', 'scripts/critique_qa.py',
                "--qas_json_fullpath="+ QAS_PATH_JSON,
                '--critic_llm_dir=' + args.critic_llm_dir, #'--output_dir=' + args.output_dir,
                '--critic_output_fullpath=' + QAS_PATH_DF])



# ----------------ANSWER_W_RAG_AND_TEST.PY-------------------- TODO
#RagOverArXiv/scripts/answer_w_rag_and_test.py
subprocess.run(['python', 'scripts/answer_w_rag_and_test.py', 
                '--pdf_or_txt=' + args.pdf_or_txt,
                '--critiqued_df_fullpath=' + QAS_PATH_DF,
                '--ragans_output_fullpath=' + QAS_PATH_DF,
                '--reader_llm_dir=' + args.reader_llm_dir,
                '--embed_model_id=' + args.embed_model_id,
                ])

#parser.add_argument('--critiqued_df_fullpath', type=str, default='./data/pdfs_ws_mrkp_test/eval_outputs/MiStralInference_txt_critiqued_qas.csv')

"""
# ----------------JUDGE_ANSWERS.PY-------------------- TODO
#RagOverArXiv/scripts/judge_answers.py
subprocess.run(['python', 'scripts/judge_answers.py', '--nquestions=' + nquestions, '--pdf_or_txt=' + pdf_or_txt, '--output_file_name=' + output_fn, '--files_path=' + files_path])


# TODO: Make paths for outputs to be passed between subprocesses and modified after each step
# OR! Just make one final path once have df to add to!
"""