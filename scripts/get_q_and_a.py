import os
from exllamav2 import *
from exllamav2.generator import *
import sys, torch
import argparse

def load_elx2_llm(model_dir="/home/mainuser/Desktop/LLMs/MixtralInference/Mixtral-8x7B-instruct-exl2"):
    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.prepare()

    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, lazy = True)

    print("Loading model...")
    model.load_autosplit(cache)

    tokenizer = ExLlamaV2Tokenizer(config)
    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
    generator.set_stop_conditions([tokenizer.eos_token_id])
    gen_settings = ExLlamaV2Sampler.Settings()

    return generator, gen_settings
def generate_q_and_a(filename, generator, gen_settings,input_dir,output_dir):
    # Read the input file
    with open(f'{input_dir}/{filename}', 'r') as f:
        paper_text = f.read()

    # Pass the input to the model
    qas = call_llm(paper_text, generator, gen_settings)

    # Write the output to a file in the 'q_and_a' directory
    with open(f'{output_dir}/qa_{filename}', 'w') as f:
        f.write(qas)

def call_llm(
    paper_text: str,
    generator: ExLlamaV2StreamingGenerator,
    settings:ExLlamaV2Sampler.Settings,
    max_new_tokens = 4096
    ):
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

parser = argparse.ArgumentParser(description='Generate questions and answers from input text')
parser.add_argument('--model_dir', type=str, default="/home/mainuser/Desktop/LLMs/MixtralInference/Mixtral-8x7B-instruct-exl2", help='Path to the model directory')
parser.add_argument('--input_dir', type=str, default='./markups_new', help='Path to the directory containing the input files')
parser.add_argument('--output_dir', type=str, default='./q_and_a', help='Path to the directory containing the output files')

def main():
    args = parser.parse_args()
    model_dir = args.model_dir
    input_dir = args.input_dir
    output_dir = args.output_dir
    generator, gen_settings = load_elx2_llm(model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Loop over the files in the 'markups_new' directory
    for filename in os.listdir(input_dir):
        print(f'Processing {filename}...')
        generate_q_and_a(filename, generator, gen_settings,input_dir,output_dir)

if __name__ == '__main__':
    main()
# TODO: see if can get title of paper and store by title (write arxiv abbrev as first line in text)
# TODO: add some timing