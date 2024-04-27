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
for output in tqdm(outputs):
    evaluations = {
        "groundedness": call_llm(question=question_groundedness_critique_prompt.format(context=output["context"], question=output["question"]), 
                                generator=generator_llm,
                                tokenizer=generator_tokenizer,settings=generator_settings,
                                max_new_tokens=1024),
        "relevance": call_llm(question=question_relevance_critique_prompt.format(question=output["question"]), 
                                generator=generator_llm,
                                tokenizer=generator_tokenizer,settings=generator_settings,
                                max_new_tokens=1024),
                    
        "standalone": call_llm(question=question_standalone_critique_prompt.format(question=output["question"]),
                                generator=generator_llm,
                                tokenizer=generator_tokenizer,settings=generator_settings,
                                max_new_tokens=1024)
    }
    try:
        for criterion, evaluation in evaluations.items():
            score, eval = (
                # int(evaluation.split("Total rating: ")[-1].strip()),
                (evaluation.split("Total rating: ")[-1].strip()),
                evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
            )
            output.update(
                {
                    f"{criterion}_score": score,
                    f"{criterion}_eval": eval,
                }
            )
    except Exception as e:
        #print("\033[91m" + f"EVALUATION:" + "\033[0m")
        #print(evaluations)
        #print("\033[91m" + f"EXCEPTION: {e}" + "\033[0m")
        continue
import pandas as pd

pd.set_option("display.max_colwidth", None)

generated_questions = pd.DataFrame.from_dict(outputs)

print("Evaluation dataset before filtering:")
display(
    generated_questions[
        [
            "question",
            "answer",
            "groundedness_score",
            "relevance_score",
            "standalone_score",
        ]
    ]
)
generated_questions['groundedness_score']=generated_questions['groundedness_score'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
generated_questions['relevance_score']=generated_questions['relevance_score'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
generated_questions['standalone_score']=generated_questions['groundedness_score'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
os.getcwd()
generated_questions.to_csv("../data/pdfs_ws_mrkp_test/generated_questions_pdf_raw.csv", index=False)
for col in ["groundedness_score", "relevance_score", "standalone_score"]:
    generated_questions[col] = generated_questions[col].fillna(generated_questions[["groundedness_score", "relevance_score", "standalone_score"]].min(axis=1))
generated_questions = generated_questions.loc[
    (generated_questions["groundedness_score"] >= 3.0)
    & (generated_questions["relevance_score"] >= 3.0)
    & (generated_questions["standalone_score"] >= 3.0)
]
print("============================================")
print("Final evaluation dataset:")
display(
    generated_questions[
        [
            "question",
            "answer",
            "groundedness_score",
            "relevance_score",
            "standalone_score",
        ]
    ]
)

# eval_dataset = datasets.Dataset.from_pandas(
#     generated_questions, split="train", preserve_index=False
# )
generated_questions.to_csv("../data/pdfs_ws_mrkp_test/generated_questions_pdf_filtered.csv", index=False)
