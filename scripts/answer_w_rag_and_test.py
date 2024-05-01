"""
`answer_w_rag_and_test.py`: A script for generating and evaluating answers using an llm model and a knowledge index.

This script contains two main functions: `answer_with_rag` and `run_rag_tests`.

`answer_with_rag` generates an answer to a given question using an llm model and a knowledge index. It returns the generated answer and the list of relevant documents retrieved from the knowledge index.

`run_rag_tests` runs tests on a given dataset using the `answer_with_rag` function. It updates the dataset with the generated answers and relevant documents, and returns the updated dataset.

This script requires the `ExLlamaV2StreamingGenerator`, `ExLlamaV2Sampler.Settings`, and `RAGPretrainedModel` objects from the `exllamav2` package, and the `FAISS` and `LangchainDocument` objects from the `faiss` and `langchain` packages respectively.

Command-line arguments:
--pdf_or_txt: Specifies the type of the input files. Should be either 'pdf' or 'txt'.
--reader_llm_dir: Path to the directory containing the chat model to use for generating the answers.
--embed_model_id: Identifier of the model to use for embedding the questions.
--critiqued_df_fullpath: Full path to the CSV file containing the critiqued questions and answers.
--ragans_output_path: Path to the directory where the output file will be saved.
--ragans_output_filename: Name of the output file.
--ragans_inout_fullpath: Full path to the input/output file. Only specify this if critic_output_dir and critic_output_filename are not specified.
--vs_dir: Directory for the VS (Visual Studio) files. This argument seems to be unused in the provided code excerpt.
--use_reranker: If True, uses a reranker model to rerank the relevant documents.
"""

import argparse
from tqdm.auto import tqdm
import sys, os
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'scripts'))
import utils

import pandas as pd

from langchain.embeddings import CacheBackedEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
store = LocalFileStore("./cache/")

from exllamav2 import *
from exllamav2.generator import *

from ragatouille import RAGPretrainedModel
from typing import Optional, List, Tuple
RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
from langchain.docstore.document import Document as LangchainDocument

RAG_PROMPT_TEMPLATE = """
<|system|>
Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.</s>
<|user|>
Context:
{context}
---
Now here is the question you need to answer.

Question: {question}
</s>
<|assistant|>
"""



def answer_with_rag(
    question: str,
    reader_llm: ExLlamaV2StreamingGenerator,
    reader_llm_settings:ExLlamaV2Sampler.Settings,
    embedding_model,
    max_new_tokens,
    knowledge_index,
    use_reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 10, #30,
    num_docs_final: int = 5,
) -> Tuple[str, List[LangchainDocument]]:
    """
    Generates an answer to a given question using an llm model and a knowledge index.

    Parameters:
    question (str): The question to answer.
    reader_llm (ExLlamaV2StreamingGenerator): The llm model to use for generating the answer.
    reader_llm_settings (ExLlamaV2Sampler.Settings): Settings for the llm model.
    embedding_model: The model to use for embedding the question.
    max_new_tokens: The maximum number of new tokens to generate.
    knowledge_index: The index to use for retrieving relevant documents.
    use_reranker (Optional[RAGPretrainedModel]): An optional reranker model to use.
    num_retrieved_docs (int, optional): The number of documents to retrieve. Defaults to 10.
    num_docs_final (int, optional): The final number of documents to use. Defaults to 5.

    Returns:
    Tuple[str, List[LangchainDocument]]: The generated answer and the list of relevant documents.
    """
    print("=> Retrieving documents...")
    embedding_vector = embedding_model.embed_query(question)
    relevant_docs = knowledge_index.similarity_search_by_vector(embedding_vector, k = num_retrieved_docs)#num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text


    if use_reranker:
        relevant_docs = RERANKER.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]


    relevant_docs = relevant_docs[:num_retrieved_docs]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

   
    reader_llm.warmup()
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    answer = reader_llm.generate_simple(final_prompt, 
    reader_llm_settings, max_new_tokens, seed = 1234)
    return answer,relevant_docs



from collections import namedtuple
def run_rag_tests(
    dataset: pd.DataFrame,
    reader_llm: ExLlamaV2StreamingGenerator,
    reader_llm_settings:ExLlamaV2Sampler.Settings,
    knowledge_index: FAISS,
    embedding_model,
    use_reranker: Optional[RAGPretrainedModel] = None,
    test_settings: str = None
):
    """
    Runs RAG tests on a given dataset and updates the dataset with the generated answers and relevant documents.

    Parameters:
    dataset (pd.DataFrame): The dataset to test on.
    reader_llm (ExLlamaV2StreamingGenerator): The llm model to use for generating the answers.
    reader_llm_settings (ExLlamaV2Sampler.Settings): Settings for the llm model.
    knowledge_index: The index to use for retrieving relevant documents.
    embedding_model: The model to use for embedding the questions.
    use_reranker (Optional[RAGPretrainedModel]): An optional reranker model to use.
    test_settings (str, optional): Optional test settings.

    Returns:
    pd.DataFrame: The updated dataset with the generated answers and relevant documents.
    """
    print("Running RAG tests...")
    dataset_copy = dataset.copy(deep=True)
    dataset_copy['retrieved_docs'] = None
    for example_row in tqdm(dataset_copy.iterrows()):
        index, example = example_row
        question = example["question"]
        if dataset_copy.loc[index,'retrieved_docs']: #already retrieved
            print(f"Continue for {index} since already processed")
            continue

        generated_answer, relevant_docs =  answer_with_rag(question, 
                                                           knowledge_index=knowledge_index, 
                                                           reader_llm=reader_llm,
                                                           reader_llm_settings=reader_llm_settings,
                                                           embedding_model=embedding_model,
                                                           max_new_tokens=512,
                                                           use_reranker = use_reranker)

        dataset_copy.at[index,'retrieved_docs'] = relevant_docs
        dataset_copy.loc[index,'true_answer'] = dataset_copy.loc[index,'answer']
        dataset_copy.loc[index,'generated_answer'] = generated_answer

        if test_settings:
            dataset_copy["test_settings"] = test_settings
       
    return dataset_copy #INDENTED ON PURPOSE, TEST RUN!




def main(vs_dir):
    reader_llm, reader_llm_settings = utils.load_elx2_llm(args.reader_llm_dir)
    embedder,core_embedding_model = utils.get_embedder(args.embed_model_id)
    vector_store = FAISS.load_local(vs_dir, embedder,allow_dangerous_deserialization=True) 
    eval_dataset = pd.read_csv(args.ragans_inout_fullpath)
    ds_rag = run_rag_tests(eval_dataset,reader_llm,reader_llm_settings,knowledge_index=vector_store,
                embedding_model=core_embedding_model,use_reranker=args.use_reranker,
                test_settings=None) #args.ragans_output_filename.split('.')[0])# max_new_tokens=1024,reranker=RERANKER,
    output_file = args.ragans_output_path+args.ragans_output_filename if not args.ragans_inout_fullpath else args.ragans_inout_fullpath
    ds_rag.to_csv(output_file, index=False)

    
parser = argparse.ArgumentParser()
parser.add_argument('--pdf_or_txt', type=str, required=True, help='pdf or txt') 
parser.add_argument('--reader_llm_dir', type=str, default="../MiStralInference", help='Path to the model directory')
parser.add_argument('--embed_model_id', type=str, default='mixedbread-ai/mxbai-embed-large-v1')
parser.add_argument('--critiqued_df_fullpath', type=str, default='./data/pdfs_ws_mrkp_test/eval_outputs/MiStralInference_txt_critiqued_qas.csv')
parser.add_argument('--ragans_output_path', type=str, default='./data/pdfs_ws_mrkp_test/eval_outputs/')
parser.add_argument('--ragans_output_filename', type=str, default='MistralQs-mxbaiEmbed-ZephyrRead-2000x200chunks-NoRerank.csv')
parser.add_argument('--ragans_inout_fullpath',type=str,default=None, help='Only specify if critic_output_dir and critic_output_filename not specified') # Or fullpath
parser.add_argument('--vs_dir', type=str, default=None)
parser.add_argument('--use_reranker', type=bool, default=False)
args = parser.parse_args()

if __name__ == '__main__':
    if not args.vs_dir:
        if args.pdf_or_txt == 'pdf':
            vs_dir = './data/rag_index_dir/pdfs'
        else:
            vs_dir ='./data/rag_index_dir/txts'
    else:
        vs_dir = args.vs_dir
    main(vs_dir=vs_dir)


