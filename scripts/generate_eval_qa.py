"""
generate_eval_qa.py

This script generates a specified number of question-answer pairs from a set of documents using a language model and saves them to a CSV file.

The script uses several classes and functions from the exllamav2 and langchain libraries, including ExLlamaV2StreamingGenerator, ExLlamaV2Sampler.Settings, and RecursiveCharacterTextSplitter. It also uses the pandas, random, tqdm, sys, and os libraries, and a utility module named utils.

The script defines two functions, call_llm and genenerate_questions, which generate an output from a given question and generate a specified number of question-answer pairs from a list of documents, respectively.

Usage example:
    python scripts/generate_eval_qa.py --nquestions=2 --pdf_or_txt='pdf' --eval_output_json='qa_pdf_2000_200_mistral.csv' --input_files_dir='./data/pdfs_ws_mrkp_test/pdfs/'
Arguments:
    --chunk_size: This argument specifies the size of the chunks to split the text into when creating Document objects. The size is measured in number of characters. The default value is 2000.

    --chunk_overlap: This argument specifies the number of characters that should overlap between consecutive chunks when splitting the text into chunks. The default value is 200.

    --nquestions: This argument specifies the number of question-answer pairs to generate. The default value is 3.

    --pdf_or_txt: This argument specifies the type of files to process. It should be either 'pdf' for PDF files or 'txt' for text files. The default value is 'txt'.

    --qagen_llm_dir: This argument specifies the path to the directory containing the language model to use for generating the question-answer pairs. The default value is '../MiStralInference'.

    --input_files_dir: This argument specifies the path to the directory containing the files to process. The default value is './data/pdfs_ws_mrkp_test/mrkps/'.

    --eval_output_dir: This argument specifies the path to the directory where the output JSON file should be saved. The default value is './data/pdfs_ws_mrkp_test/eval_outputs/'.

"""

from typing import List
from exllamav2 import *
from exllamav2.generator import *
from langchain.text_splitter import RecursiveCharacterTextSplitter

#import pandas as pd
import json
import random
from tqdm import tqdm
import sys, os
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'scripts'))
import utils

from argparse import ArgumentParser




def genenerate_questions(n_generations: int, docs: List[str],llm,llm_settings):
    """
    Generate a specified number of question-answer pairs from a list of documents.

    Parameters:
    n_generations (int): The number of question-answer pairs to generate.
    docs (List[str]): The list of documents to generate the question-answer pairs from.
    llm: The language model to use for generating the question-answer pairs.
    llm_settings: The settings to use for the language model.

    Returns:
    List[dict]: A list of dictionaries, each containing a context, a question, an answer, and a source document.
    """
    print(f"Generating {n_generations} QA couples...")
    QA_generation_prompt = """
    Your task is to write a deep factual or conceptual question and an answer given a context.
    Your deep question should be unambigiously answerable from the context.
    Your deep question should be formulated in the same style as questions people reading advanced LLM papers would ask.
    This means that your question MUST NOT mention something like "according to the passage" or "context".

    Provide your answer as follows:

    Output:::
    Deep question: (your deep question)
    Answer: (your answer to the deep question)

    Now here is the context.

    Context: {context}\n
    Output:::"""
    

    outputs = []
    for sampled_context in tqdm(random.sample(docs, n_generations)):
        output_QA_couple = utils.call_llm(question=QA_generation_prompt.format(context=sampled_context.page_content), 
                                    generator=llm,
                                    settings=llm_settings,
                                    max_new_tokens=1024)
        try:
            question = output_QA_couple.split("Deep question: ")[-1].split("Answer: ")[0]
            answer = output_QA_couple.split("Answer: ")[-1]
            outputs.append(
                {
                    "context": sampled_context.page_content,
                    "question": question,
                    "answer": answer,
                    "source_doc": sampled_context.metadata["title"],
                }
            )
        except Exception as e:
            print(f"Exception: {e}")
            continue
    return outputs


def main(llm,llm_settings,eval_output_json):
    """
    Qenerate question-answer pairs from a set of documents and save them to a CSV file.

    Parameters:
    llm: The language model to use for generating the question-answer pairs.
    llm_settings: The settings to use for the language model.
    eval_output_json (str): The name of the CSV file to save the question-answer pairs to.
    Returns:
        None
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap = args.chunk_overlap,
        length_function=len,
    )
  
    if args.pdf_or_txt == 'txt':
        docs = utils.get_docs_from_txt(args.input_files_dir,text_splitter=text_splitter)
    elif args.pdf_or_txt == 'pdf':
        docs = utils.get_docs_from_pdf(args.input_files_dir,text_splitter=text_splitter)
    else:
        print('Please specify pdf or txt')
        exit(1)
    outputs = genenerate_questions(n_generations=args.nquestions, docs=docs,llm=llm,llm_settings=llm_settings)
    #pd.DataFrame(outputs).to_csv(eval_output_json, index=False)
    print(f"Writing generated questions to {eval_output_json}")
    with open(eval_output_json, 'w') as f:
        json.dump(outputs, f)
    #return pd.DataFrame(outputs)


parser = ArgumentParser()
parser.add_argument('--chunk_size', type=int, default=2000)
parser.add_argument('--chunk_overlap', type=int, default=200)
parser.add_argument('--nquestions', type=int, default=10)
parser.add_argument('--pdf_or_txt', type=str, default='txt')
parser.add_argument('--qagen_llm_dir', type=str, default="../MiStralInference", help='Path to the model directory')
parser.add_argument('--input_files_dir', type=str, default='./data/pdfs_ws_mrkp_test/mrkps/')
parser.add_argument('--eval_output_dir', type=str, default='./data/pdfs_ws_mrkp_test/eval_outputs/') # Can specify these 
parser.add_argument('--eval_output_json', type=str, default='qa_couples.json')                   # two args
parser.add_argument('--eval_output_fullpath',type=str,default=None, help='Only specify if eval_output_dir and eval_output_filename not specified') # Or fullpath
args = parser.parse_args()
if __name__ == "__main__":
    generator, gen_settings = utils.load_elx2_llm(model_dir=args.qagen_llm_dir)
    eval_output_json =os.path.join(args.eval_output_dir,  args.eval_output_json) if not args.eval_output_fullpath else args.eval_output_fullpath
    main(llm=generator,llm_settings=gen_settings,eval_output_json = eval_output_json)