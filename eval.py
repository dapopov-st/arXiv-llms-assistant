import os
import sys
import subprocess
import json
import argparse
from colorama import Fore, Style

# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to the config file')

# Parse the --config argument
args, unknown = parser.parse_known_args()

# Load the configuration file
with open(args.config, 'r') as f:
    config = json.load(f)

# Add the rest of the arguments
parser.add_argument('--pdf_or_txt', type=str, default=config['pdf_or_txt'])
parser.add_argument('--chunk_size', type=int, default=config['chunk_size'])
parser.add_argument('--chunk_overlap', type=int, default=config['chunk_overlap'])
parser.add_argument('--nquestions', type=int, default=config['nquestions'])
parser.add_argument('--eval_output_dir', type=str, default=config['eval_output_dir'])
parser.add_argument('--embed_model_id', type=str, default=config['embed_model_id'])
parser.add_argument('--qagen_llm_dir', type=str, default=config['qagen_llm_dir'])
parser.add_argument('--critic_llm_dir', type=str, default=config['critic_llm_dir'])
parser.add_argument('--reader_llm_dir', type=str, default=config['reader_llm_dir'], help='Path to the reader directory')
parser.add_argument('--judge_llm_dir', type=str, default=config['judge_llm_dir'], help='Path to the judge directory')
parser.add_argument('--use_reranker', type=bool, default=config['use_reranker'])
#parser.add_argument('--use_reranker', action='store_true',default=config['use_reranker'])
# Parse all arguments
args = parser.parse_args()
print(args)#; exit(1)
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
RERANK = 'Rerank' if args.use_reranker else 'NoRerank'



#QAS_NAME = "_".join([PDF_OR_TXT,CHUNK_SIZE, CHUNK_OVERLAP ,EMBED_MODEL,CRITIC_LLM,QAGEN_LLM])
QAS_NAME = "_".join([PDF_OR_TXT,CHUNK_SIZE, CHUNK_OVERLAP ,EMBED_MODEL,CRITIC_LLM,QAGEN_LLM,RERANK])

QAS_NAME_JSON, QAS_NAME_DF = QAS_NAME+'.json',QAS_NAME+'.csv'
QAS_PATH_JSON, QAS_PATH_DF = os.path.join(args.eval_output_dir,QAS_NAME_JSON),os.path.join(args.eval_output_dir,QAS_NAME_DF)


# ----------------GET_VECTOR_STORE.PY--------------------
print(f'{Fore.BLUE}Running get_vector_store.py {Style.RESET_ALL}')
subprocess.run(['python', 'scripts/get_vector_store.py', '--pdf_or_txt=' + args.pdf_or_txt, '--files_path=' + FILES_PATH, 
                '--chunk_size=' + str(args.chunk_size), '--chunk_overlap=' + str(args.chunk_overlap)])

print(f"{Fore.BLUE}Generated vector store and stored at {INDEX_PATH}{Style.RESET_ALL}")


# ----------------GENERATE_EVAL_QA.PY-------------------- Generated JSON
print(f'{Fore.BLUE}Running generate_eval_qa.py {Style.RESET_ALL}')
subprocess.run(['python', 'scripts/generate_eval_qa.py', '--nquestions=' + str(args.nquestions), '--pdf_or_txt=' + args.pdf_or_txt, 
               '--chunk_size=' + str(args.chunk_size), '--chunk_overlap=' + str(args.chunk_overlap), 
                '--input_files_dir=' + FILES_PATH, #'--eval_output_path=' + args.eval_output_path,
                '--qagen_llm_dir=' + args.qagen_llm_dir,  '--eval_output_fullpath=' + QAS_PATH_JSON])

# ----------------CRITIQUE_QA.PY-------------------- 
print(f'{Fore.BLUE}Running critique_qa.py {Style.RESET_ALL}')
subprocess.run(['python', 'scripts/critique_qa.py',
                "--qas_json_fullpath="+ QAS_PATH_JSON,
                '--critic_llm_dir=' + args.critic_llm_dir, #'--output_dir=' + args.output_dir,
                '--critic_output_fullpath=' + QAS_PATH_DF])

# ----------------ANSWER_W_RAG_AND_TEST.PY--------------------
# Will load from local vs_dir (pdf or txt) @ Index Path vs_dir arg is passed to script
print(f'{Fore.BLUE}Running answer_w_rag_and_test.py {Style.RESET_ALL}')
if args.use_reranker:
    subprocess.run(['python', 'scripts/answer_w_rag_and_test.py', 
                '--pdf_or_txt=' + args.pdf_or_txt,
                '--ragans_inout_fullpath=' + QAS_PATH_DF,
                '--reader_llm_dir=' + args.reader_llm_dir,
                '--embed_model_id=' + args.embed_model_id,
                '--use_reranker'
                ])
else:
    subprocess.run(['python', 'scripts/answer_w_rag_and_test.py', 
                    '--pdf_or_txt=' + args.pdf_or_txt,
                    '--ragans_inout_fullpath=' + QAS_PATH_DF,
                    '--reader_llm_dir=' + args.reader_llm_dir,
                    '--embed_model_id=' + args.embed_model_id,
                    ])

# ----------------JUDGE_ANSWERS.PY-------------------- 
print(f'{Fore.BLUE}Running judge_answers.py {Style.RESET_ALL}')
subprocess.run(['python', 'scripts/judge_answers.py', '--ragans_inout_fullpath=' + QAS_PATH_DF, 
                '--judge_llm_dir=' + args.judge_llm_dir])
