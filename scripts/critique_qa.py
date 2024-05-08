"""
`critique_qa.py`: A script for generating critiques for question-answer pairs using a chat model.

This script uses the `gen_critiques` and `cleanup_critique` functions to generate and clean up the critiques. The critiques are saved to a CSV file.

Command-line arguments:
    --qas_json_fullpath`: Full path to the JSON file containing the question-answer pairs.
    --critic_output_dir`: Path to the directory where the output file will be saved. Defaults to './data/pdfs_ws_mrkp_test/eval_outputs/'.
    --critic_llm_dir`: Path to the directory containing the chat model. Defaults to '../MiStralInference'.
    --critic_output_filename`: Name of the output file. Defaults to 'critiqued_qas.csv'.
    --critic_output_fullpath`: Full path to the output file. Only specify this if `critic_output_dir` and `critic_output_filename` are not specified. Defaults to None.

This script requires the `ExLlamaV2StreamingGenerator` and `ExLlamaV2Sampler.Settings` objects from the `exllamav2` package.
"""

from argparse import ArgumentParser
from tqdm.auto import tqdm
import pandas as pd
import json
from typing import Dict

import sys, os
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'scripts'))

import utils


def gen_critiques(critic_llm_dir:str, qas_json_fullpath:str)->pd.DataFrame:
    """
    Generates critiques for each question-answer pair in a given JSON file using a chat model.

    Parameters:
    critic_llm_dir (str): Path to the directory containing the chat model.
    qas_json_fullpath (str): Full path to the JSON file containing the question-answer pairs.

    Returns:
    pd.DataFrame: A DataFrame containing the question-answer pairs and their critiques.
    """
    with open(qas_json_fullpath, 'r') as f:
        qas = json.load(f)
    generator_llm, generator_settings = utils.load_elx2_llm(critic_llm_dir)

    question_groundedness_critique_prompt = """
    You will be given a context and a question.
    Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
    Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

    Provide your answer as follows:

    Answer:::
    Evaluation: (your rationale for the rating, as a text)
    Total rating: (your rating, as a number between 1 and 5)

    You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

    Now here are the question and context.

    Question: {question}\n
    Context: {context}\n
    Answer::: """

    question_relevance_critique_prompt = """
    You will be given a question.
    Your task is to provide a 'total rating' representing how useful this question can be to machine learning developers building NLP applications with the Hugging Face ecosystem.
    Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

    Provide your answer as follows:

    Answer:::
    Evaluation: (your rationale for the rating, as a text)
    Total rating: (your rating, as a number between 1 and 5)

    You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

    Now here is the question.

    Question: {question}\n
    Answer::: """

    question_standalone_critique_prompt = """
    You will be given a question.
    Your task is to provide a 'total rating' representing how context-independant this question is.
    Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
    For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
    The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

    For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independant from the context.

    Provide your answer as follows:

    Answer:::
    Evaluation: (your rationale for the rating, as a text)
    Total rating: (your rating, as a number between 1 and 5)

    You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

    Now here is the question.

    Question: {question}\n
    Answer::: """
    print("Generating critique for each QA couple...")
    for qa in tqdm(qas):
        evaluations = {
            "groundedness": utils.call_llm(question=question_groundedness_critique_prompt.format(context=qa["context"], question=qa["question"]), 
                                    generator=generator_llm,
                                    settings=generator_settings,
                                    max_new_tokens=1024),
            "relevance": utils.call_llm(question=question_relevance_critique_prompt.format(question=qa["question"]), 
                                    generator=generator_llm,
                                    settings=generator_settings,
                                    max_new_tokens=1024),
                        
            "standalone": utils.call_llm(question=question_standalone_critique_prompt.format(question=qa["question"]),
                                    generator=generator_llm,
                                    settings=generator_settings,
                                    max_new_tokens=1024)
        }
        try:
            for criterion, evaluation in evaluations.items():
                score, eval = (
                    # int(evaluation.split("Total rating: ")[-1].strip()),
                    (evaluation.split("Total rating: ")[-1].strip()),
                    evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
                )
                qa.update(
                    {
                        f"{criterion}_score": score,
                        f"{criterion}_eval": eval,
                    }
                )
        except Exception as e:
            #print("\033[91m" + f"EXCEPTION: {e}" + "\033[0m")
            continue
    cleaned_critiques=cleanup_critique(json_outputs=qas)
    return cleaned_critiques
 

def cleanup_critique(json_outputs:Dict)->pd.DataFrame:
    """
    Cleans up the critiques generated by the `gen_critiques` function.

    Parameters:
    json_outputs (list): The output from the `gen_critiques` function.

    Returns:
    pd.DataFrame: A DataFrame containing the cleaned-up critiques.
    """
    generated_questions = pd.DataFrame.from_dict(json_outputs)
    generated_questions['groundedness_score']=generated_questions['groundedness_score'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    generated_questions['relevance_score']=generated_questions['relevance_score'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    generated_questions['standalone_score']=generated_questions['groundedness_score'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

    for col in ["groundedness_score", "relevance_score", "standalone_score"]:
        generated_questions[col] = generated_questions[col].fillna(generated_questions[["groundedness_score", "relevance_score", "standalone_score"]].min(axis=1))
    generated_questions = generated_questions.loc[
        (generated_questions["groundedness_score"] >= 3.0)
        & (generated_questions["relevance_score"] >= 3.0)
        & (generated_questions["standalone_score"] >= 3.0)
    ]
    # print("============================================")
    # print("Final evaluation dataset head(2):")
    # print(generated_questions.head(2).to_string())
    return generated_questions


def main():
    critiques = gen_critiques(critic_llm_dir=args.critic_llm_dir, qas_json_fullpath=args.qas_json_fullpath)
    #llm = args.critic_llm_dir.split("/")[-1]
    output_file = args.critic_output_dir+args.critic_output_filename if args.critic_output_fullpath is None else args.critic_output_fullpath
    critiques.to_csv(output_file, index=False)
    print(f"Generated critiques saved to {output_file}")


parser = ArgumentParser()
parser.add_argument("--qas_json_fullpath", type=str, required=True)
parser.add_argument("--critic_output_dir", type=str, default="./data/pdfs_ws_mrkp_test/eval_outputs/")
parser.add_argument('--critic_llm_dir', type=str, default="../MiStralInference", help='Path to the model directory')
parser.add_argument('--critic_output_filename', type=str, default='critiqued_qas.csv')
parser.add_argument('--critic_output_fullpath',type=str,default=None, help='Only specify if critic_output_dir and critic_output_filename not specified') # Or fullpath

args = parser.parse_args()
if __name__ == "__main__":
    main()
    