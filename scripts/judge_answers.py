
from exllamav2 import *
from exllamav2.generator import *

judge_config = ExLlamaV2Config()
judge_config.model_dir = "../PrometheusEval"
#judge_config.model_dir = '/home/mainuser/Desktop/LLMs/Mixtral4bit'
judge_config.prepare()

judge_model = ExLlamaV2(judge_config)
cache = ExLlamaV2Cache(judge_model, lazy = True)

print("Loading model...")
judge_model.load_autosplit(cache)

judge_tokenizer = ExLlamaV2Tokenizer(judge_config)
judge_llm = ExLlamaV2StreamingGenerator(judge_model, cache, judge_tokenizer)
#judge_llm.set_stop_conditions([judge_tokenizer.eos_token_id])
judge_settings = ExLlamaV2Sampler.Settings()
judge_settings.temperature = 1.0
# judge_settings.top_k = 30
# judge_settings.top_p = 0.8
# judge_settings.token_repetition_penalty = 1.03



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

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage


evaluation_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
    ]
)


import re

def evaluate_answers(
    answer_path: str,
    eval_chat_model:ExLlamaV2StreamingGenerator,
    settings:ExLlamaV2Sampler.Settings,
    evaluation_prompt: str
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
        settings, num_tokens=1024, seed = 1234) #max_new_tokens=1024,
        feedback = re.search(r'###Feedback:\s*(.*)',eval_result,re.DOTALL).group(1)
        try:
            #score = re.search(r'(\d+)', feedback).group(1)
            score = re.search(r'overall score is (\d)', feedback).group(1)
        except AttributeError:
            score = 'NaN'
        answers.loc[index,f"eval_score"] = score
        answers.loc[index,f"eval_feedback"] = feedback
        print(f'Score: {score}')
        print(f'Feedback: {feedback}')
    return answers #INDENTED ON PURPOSE, TEST RUN!
        # with open(answer_path, "w") as f:
        #     json.dump(answers, f)

def main():
    temp=evaluate_answers(answer_path='../data/pdfs_ws_mrkp_test/MistralQs-mxbai_embed-ZephyrRead-2000x200chunks-NoRerank.csv',
                    eval_chat_model=judge_llm,settings=judge_settings,evaluation_prompt=EVALUATION_PROMPT)
    temp.to_csv("../data/pdfs_ws_mrkp_test/MistralQs-mxbai_embed-ZephyrRead-2000x200chunks-NoRerank-Evaluated.csv", index=False)
    import matplotlib.pyplot as plt
    temp.eval_score.sort_values().hist()
    plt.title("Pdf-MistralQs-mxbai_embed-ZephyrRead-2000x200chunks-NoRerank");
    plt.savefig('../data/pdfs_ws_mrkp_test/Pdf-MistralQs-mxbai_embed-ZephyrRead-2000x200chunks-NoRerank-Evaluated.png')
