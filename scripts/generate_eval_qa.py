"""
generate_eval_qa.py

This script generates a specified number of question-answer pairs from a set of documents using a language model and saves them to a CSV file.

The script uses several classes and functions from the exllamav2 and langchain libraries, including ExLlamaV2StreamingGenerator, ExLlamaV2Sampler.Settings, and RecursiveCharacterTextSplitter. It also uses the pandas, random, tqdm, sys, and os libraries, and a utility module named utils.

The script defines two functions, call_llm and genenerate_questions, which generate an output from a given question and generate a specified number of question-answer pairs from a list of documents, respectively.

Usage example:
    python scripts/generate_eval_qa.py --nquestions=2 --pdf_or_txt='pdf' --output_file_name='qa_pdf_2000_200_mistral.csv' --files_path='./data/pdfs_ws_mrkp_test/pdfs/'
Arguments:
    --chunk_size: This argument specifies the size of the chunks to split the text into when creating Document objects. The size is measured in number of characters. The default value is 2000.

    --chunk_overlap: This argument specifies the number of characters that should overlap between consecutive chunks when splitting the text into chunks. The default value is 200.

    --nquestions: This argument specifies the number of question-answer pairs to generate. The default value is 3.

    --pdf_or_txt: This argument specifies the type of files to process. It should be either 'pdf' for PDF files or 'txt' for text files. The default value is 'txt'.

    --llm_dir: This argument specifies the path to the directory containing the language model to use for generating the question-answer pairs. The default value is '../MiStralInference'.

    --files_path: This argument specifies the path to the directory containing the files to process. The default value is './data/pdfs_ws_mrkp_test/mrkps/'.

    --output_path: This argument specifies the path to the directory where the output CSV file should be saved. The default value is './data/pdfs_ws_mrkp_test/dfs/'.

    --output_file_name: This argument specifies the name of the CSV file to save the question-answer pairs to. The default value is 'qa_couples.csv'.
"""

from typing import List
from exllamav2 import *
from exllamav2.generator import *
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pandas as pd
import random
from tqdm import tqdm
import sys, os
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'scripts'))
import utils

from argparse import ArgumentParser

def call_llm(
    question: str,
    generator: ExLlamaV2StreamingGenerator,
    settings:ExLlamaV2Sampler.Settings,
    max_new_tokens = 512
) -> str:
    """
    Generate an output from a given question using a specified generator and settings.

    Parameters:
    question (str): The question to generate an output from.
    generator (ExLlamaV2StreamingGenerator): The generator to use for generating the output.
    settings (ExLlamaV2Sampler.Settings): The settings to use for the generator.
    max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.

    Returns:
    Tuple[str, List[LangchainDocument]]: The generated output and a list of LangchainDocument objects.
    """
    max_new_tokens = max_new_tokens

    generator.warmup()
    output = generator.generate_simple(f"<s>[INST] {question} [/INST]", settings, max_new_tokens, seed = 1234)
    return output




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
        output_QA_couple = call_llm(question=QA_generation_prompt.format(context=sampled_context.page_content), 
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


def main(llm,llm_settings,output_file_name):
    """
    Qenerate question-answer pairs from a set of documents and save them to a CSV file.

    Parameters:
    llm: The language model to use for generating the question-answer pairs.
    llm_settings: The settings to use for the language model.
    output_file_name (str): The name of the CSV file to save the question-answer pairs to.
    Returns:
        None
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap = args.chunk_overlap,
        length_function=len,
    )
  
    if args.pdf_or_txt == 'txt':
        docs = utils.get_docs_from_txt(args.files_path,text_splitter=text_splitter)
    elif args.pdf_or_txt == 'pdf':
        docs = utils.get_docs_from_pdf(args.files_path,text_splitter=text_splitter)
    else:
        print('Please specify pdf or txt')
        exit(1)
    outputs = genenerate_questions(n_generations=args.nquestions, docs=docs,llm=llm,llm_settings=llm_settings)
    pd.DataFrame(outputs).to_csv(output_file_name, index=False)
    #return pd.DataFrame(outputs)


parser = ArgumentParser()
parser.add_argument('--chunk_size', type=int, default=2000)
parser.add_argument('--chunk_overlap', type=int, default=200)
parser.add_argument('--nquestions', type=int, default=3)
parser.add_argument('--pdf_or_txt', type=str, default='txt')
parser.add_argument('--llm_dir', type=str, default="../MiStralInference", help='Path to the model directory')
parser.add_argument('--files_path', type=str, default='./data/pdfs_ws_mrkp_test/mrkps/')
parser.add_argument('--output_path', type=str, default='./data/pdfs_ws_mrkp_test/dfs/')
parser.add_argument('--output_file_name', type=str, default='qa_couples.csv')
args = parser.parse_args()
if __name__ == "__main__":
    generator, gen_settings = utils.load_elx2_llm(model_dir=args.llm_dir)
    output_file_name =os.path.join(args.output_path,  args.output_file_name)
    main(llm=generator,llm_settings=gen_settings,output_file_name = output_file_name)