import argparse
import re
from tqdm.auto import tqdm
import sys, os
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'scripts'))
import utils

import pandas as pd
import matplotlib.pyplot as plt

from exllamav2 import *
from exllamav2.generator import *

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage



def evaluate_answers(
    answer_path: str,
    eval_chat_model:ExLlamaV2StreamingGenerator,
    settings:ExLlamaV2Sampler.Settings,
    evaluation_prompt: str,
    verbose: bool = False
) -> None:
    """Evaluates generated answers. Modifies the given answer file in place for better checkpointing."""
    answers = []
    if os.path.isfile(answer_path):  # load previous generations if they exist
        answers = pd.read_csv(answer_path)
    for example_row in tqdm(answers.iterrows()):
        index, example = example_row
        if f"eval_score" in example:
            continue

        eval_prompt = evaluation_prompt.format(
            instruction=example["question"],
            response=example["generated_answer"],
            reference_answer=example["true_answer"],
        )

        eval_chat_model.warmup()
        
        eval_result = eval_chat_model.generate_simple(eval_prompt, 
        settings, num_tokens=1024, seed = 1234) 
        feedback = re.search(r'###Feedback:\s*(.*)',eval_result,re.DOTALL).group(1)
        try:
            #score = re.search(r'(\d+)', feedback).group(1)
            score = re.search(r'overall score is (\d)', feedback).group(1)
        except AttributeError:
            score = 'NaN'
        answers.loc[index,f"eval_score"] = score
        answers.loc[index,f"eval_feedback"] = feedback
        if verbose:
            print(f'Score: {score}')
            print(f'Feedback: {feedback}')
    return answers #INDENTED ON PURPOSE, TEST RUN!
        

def main():

    EVALUATION_PROMPT = """###Task Description:
    An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
    1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
    2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
    3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
    4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

    ###The instruction to evaluate:
    {instruction}

    ###Response to evaluate:
    {response}

    ###Reference Answer (Score 5):
    {reference_answer}

    ###Score Rubrics:
    [Is the response correct, accurate, and factual based on the reference answer?]
    Score 1: The response is completely incorrect, inaccurate, and/or not factual.
    Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
    Score 3: The response is somewhat correct, accurate, and/or factual.
    Score 4: The response is mostly correct, accurate, and factual.
    Score 5: The response is completely correct, accurate, and factual.

    ###Feedback:"""

    judge_llm, judge_settings = utils.load_elx2_llm(args.judge_llm_dir)
    judge_settings.temperature = 1.0
    evaluation_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
    ]
    )


    judged_answers_df=evaluate_answers(answer_path=args.ragans_inout_fullpath,
                    eval_chat_model=judge_llm,settings=judge_settings,evaluation_prompt=evaluation_prompt_template)
    judged_answers_df.to_csv(args.ragans_inout_fullpath, index=False)
    
    judged_answers_df.eval_score.sort_values().hist()
    plt.title(args.ragans_inout_fullpath.split('/')[-1])
    plt.savefig(args.ragans_inout_fullpath.replace('.csv','.png'))

parser = argparse.ArgumentParser()
parser.add_argument('--ragans_inout_fullpath', type=str, required=True, help='pdf or txt')
parser.add_argument('--judge_llm_dir', type=str, default="../PrometheusEval", help='Path to the model directory')
args = parser.parse_args()
if __name__ == '__main__':
    main()