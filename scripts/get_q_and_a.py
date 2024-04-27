"""
This script generates in-depth questions and answers about scientific papers using the ExLlamaV2 language model.

The script reads the text of a paper from a file, generates questions and answers about the paper using the language model, and writes the questions and answers to a file in a specified directory. It also fetches the title of the paper from the arXiv API.

The script takes three command-line arguments:
    --model_dir: The directory where the ExLlamaV2 model files are located. Defaults to "../MixtralInference/Mixtral-8x7B-instruct-exl2".
    --input_dir: The directory where the input files are located. Defaults to './markups_new'.
    --output_dir: The directory where the output files should be written. Defaults to './q_and_a'.

Usage:
    python get_q_and_a.py --model_dir <model_dir> --input_dir <input_dir> --output_dir <output_dir>

Example:
    python get_q_and_a.py --model_dir "../MixtralInference/Mixtral-8x7B-instruct-exl2" --input_dir "./markups_new" --output_dir "./q_and_a"
"""


from exllamav2 import *
from exllamav2.generator import *


import os
import argparse
import logging
from datetime import datetime
import argparse
import time
# Set up basic logging
logging.basicConfig(filename='./logs/get_q_and_a.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s')

from utils import get_title, move_processed_files,load_elx2_llm


def call_llm(
    paper_text: str,
    generator: ExLlamaV2StreamingGenerator,
    settings:ExLlamaV2Sampler.Settings,
    max_new_tokens = 4096
    ):
    """
    Generates in-depth questions and answers about a given paper using the ExLlamaV2 language model.

    This function creates a prompt asking for 10 in-depth questions about the method and results of the paper. It then passes the prompt to the language model and returns the generated text.

    Args:
        paper_text (str): The text of the paper.
        generator (ExLlamaV2StreamingGenerator): The ExLlamaV2StreamingGenerator instance to use for text generation.
        settings (ExLlamaV2Sampler.Settings): The settings to use for text generation.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 4096.

    Returns:
        str: The generated text, which should contain the questions and answers about the paper.
    """

    prompt = f"""Design 10 in-depth questions about the method proposed in the paper below as well as its results. 
    Avoid simple questions.  Provide answers along with the questions in Question-Answer format.  The paper is below:
    {paper_text}
    Once again, design 10 in-depth questions about the method proposed in the paper above as well as its results. 
    Avoid simple questions.  Provide answers along with the questions in Question-Answer format.
    """
    end_instruction_idx = -1
    generator.warmup()
    output = generator.generate_simple(f"<s>[INST] {prompt} [/INST]", settings, max_new_tokens, seed = 1234)
    end_instruction_idx = output.find('[/INST]')

    return output  if end_instruction_idx == -1 else output[end_instruction_idx+7:]


def generate_q_and_a(filename, generator, gen_settings,input_dir):
    """
    Generates questions and answers for a given paper using a language model and returns them.

    This function reads the text of a paper from a file and generates questions and answers about the paper using a language model.

    Args:
        filename (str): The name of the file containing the paper text.
        generator (ExLlamaV2StreamingGenerator): The ExLlamaV2StreamingGenerator instance to use for text generation.
        gen_settings (ExLlamaV2Sampler.Settings): The settings to use for text generation.
        input_dir (str): The directory where the input file is located.

    Returns:
        Q and A generated for the paper.
    """

    with open(f'{input_dir}/{filename}', 'r') as f:
        paper_text = f.read()

    start = time.time()
    qas = call_llm(paper_text, generator, gen_settings)
    print(f"Time to generate Q and A: {time.time()-start:.2f} seconds")

    return qas
    
def write_q_and_a(filename,qas,output_dir):
    """
    Writes questions and answers to a file and logs the operation.

    This function takes a filename, questions and answers (qas), and an output directory. It extracts the arXiv ID from the filename, gets the title of the paper, and writes the title and qas to a file in the output directory. If the length of qas is less than 100, it logs a message.

    Args:
        filename (str): The name of the file containing the paper text.
        qas (str): The generated questions and answers.
        output_dir (str): The directory where the output file should be written.

    Returns:
        None
    """
    arxiv_id = filename.split('_')[0]
    title = get_title(arxiv_id)
    with open(f'{output_dir}/qa_{arxiv_id}_{title[:50].replace(' ','_')}.txt', 'w') as f:
        f.write(title+'\n'+qas)
        logging.info(f'Successfully writing Q and A for {title} on {datetime.now()}')
    if len(qas) < 100:
        logging.info(f'Fewer than 100 characters returned for {filename} on {datetime.now()}')


parser = argparse.ArgumentParser(description='Generate questions and answers from input text')
parser.add_argument('--model_dir', type=str, default="../MixtralInference/Mixtral-8x7B-instruct-exl2", help='Path to the model directory')
parser.add_argument('--input_dir', type=str, default='./data/markups_new', help='Path to the directory containing the input files')
parser.add_argument('--output_dir', type=str, default='./data/q_and_a', help='Path to the directory containing the output files')
parser.add_argument('--processed_dir', type=str, default='.data/markups_new_processed', help='Path to the directory containing markups for processed files')

def main():
    start_time = time.time()
    args = parser.parse_args()
    model_dir = args.model_dir
    input_dir = args.input_dir
    output_dir = args.output_dir
    processed_dir = args.processed_dir
    start_load = time.time()
    generator, gen_settings = load_elx2_llm(model_dir)
    print(f"Time to load model: {time.time()-start_load:.2f} seconds")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Loop over the files in the 'markups_new' directory
    for filename in os.listdir(input_dir):
        print(f'Processing {filename}...')
        qas=generate_q_and_a(filename, generator, gen_settings,input_dir)
        write_q_and_a(filename,qas,output_dir)
        # Move processed files to 'markups_new_processed' directory
        move_processed_files(input_dir, processed_dir, filename)
       
    print(f"Total time: {time.time()-start_time:.2f} seconds for {len(os.listdir(input_dir))} articles")

if __name__ == '__main__':
    main()
