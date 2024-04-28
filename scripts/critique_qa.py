from argparse import ArgumentParser
from tqdm.auto import tqdm
import pandas as pd
pd.set_option("display.max_colwidth", None)
import json

import sys, os
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'scripts'))

import utils


def gen_critiques(llm_dir, qas_dir):
    with open(qas_dir, 'r') as f:
        qas = json.load(f)
    generator_llm, generator_settings = utils.load_elx2_llm(llm_dir)

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
 

def cleanup_critique(json_outputs):
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
    print("============================================")
    print("Final evaluation dataset head(2):")
    print(generated_questions.head(2).to_string())
    return generated_questions

# eval_dataset = datasets.Dataset.from_pandas(
#     generated_questions, split="train", preserve_index=False
# )

def main():
    critiques = gen_critiques(llm_dir=args.llm_dir, qas_dir=args.qas_dir)
    llm = args.llm_dir.split("/")[-1]
    output_file = args.output_dir+llm+'_'+args.pdf_or_txt+'_'+"critiqued_qas.csv"
    critiques.to_csv(output_file, index=False)
    print(f"Generated critiques saved to {output_file}")
parser = ArgumentParser()
parser.add_argument("--qas_dir", type=str, required=True)
parser.add_argument('--pdf_or_txt', type=str, required=True)
parser.add_argument("--output_dir", type=str, default="./data/pdfs_ws_mrkp_test/eval_outputs/")
parser.add_argument('--llm_dir', type=str, default="../MiStralInference", help='Path to the model directory')
args = parser.parse_args()
if __name__ == "__main__":
    main()
    #parser