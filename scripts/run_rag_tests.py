from collections import namedtuple
def run_rag_tests(
    dataset: pd.DataFrame,
    llm: ExLlamaV2StreamingGenerator,
    knowledge_index: VectorStore,
    #output_file: str,
    reranker: Optional[RAGPretrainedModel] = None,
    verbose: Optional[bool] = False,
    test_settings: Optional[str] = None,  # To document the test settings used
):
    """Runs RAG tests on the given dataset and saves the results to the given output file."""

    dataset_copy = dataset.copy(deep=True)
    dataset_copy['retrieved_docs'] = None
    for example_row in tqdm(dataset_copy.iterrows()):
        index, example = example_row
        question = example["question"]
        if dataset_copy.loc[index,'retrieved_docs']: #already retrieved
            print(f"Continue for {index} since already processed")
            continue

        generated_answer, relevant_docs =  answer_with_rag(question, knowledge_index=knowledge_index, generator=llm,settings=reader_settings,max_new_tokens=512,reranker = reranker)
        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f'True answer: {example["answer"]}')
        dataset_copy.at[index,'retrieved_docs'] = relevant_docs
        dataset_copy.loc[index,'true_answer'] = dataset_copy.loc[index,'answer']
        dataset_copy.loc[index,'generated_answer'] = generated_answer


        if test_settings:
            dataset_copy["test_settings"] = test_settings
    return dataset_copy #INDENTED ON PURPOSE, TEST RUN!
ds_rag = run_rag_tests(eval_dataset,reader_llm,vector_store,reranker = None,test_settings='MistralQs-all_MiniLM_L6_v2Embed-ZephyrRead-2000x200chunks-NoRerank')
temp=evaluate_answers(answer_path='../data/pdfs_ws_mrkp_test/MistralQs-mxbai_embed-ZephyrRead-2000x200chunks-NoRerank.csv',
                 eval_chat_model=judge_llm,settings=judge_settings,evaluation_prompt=EVALUATION_PROMPT)
temp.to_csv("../data/pdfs_ws_mrkp_test/MistralQs-mxbai_embed-ZephyrRead-2000x200chunks-NoRerank-Evaluated.csv", index=False)
import matplotlib.pyplot as plt
temp.eval_score.sort_values().hist()
plt.title("Pdf-MistralQs-mxbai_embed-ZephyrRead-2000x200chunks-NoRerank");
plt.savefig('../data/pdfs_ws_mrkp_test/Pdf-MistralQs-mxbai_embed-ZephyrRead-2000x200chunks-NoRerank-Evaluated.png')
