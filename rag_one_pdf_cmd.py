
import subprocess
from argparse import ArgumentParser
from tqdm.auto import tqdm
import sys, os, shutil, glob
from scripts.utils import load_elx2_llm, get_embedder
from colorama import Fore, Style

import pandas as pd
import streamlit as st

from langchain.embeddings import CacheBackedEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
store = LocalFileStore("./cache/")

from exllamav2 import *
from exllamav2.generator import *

from langchain.docstore.document import Document as LangchainDocument

from ragatouille import RAGPretrainedModel
from typing import Optional, List, Tuple
CUDA_VISIBLE_DEVICES=1
RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
READER_DIR = '../MixtralInference/Mixtral-8x7B-instruct-exl2'

# READER_LLM, READER_LLM_SETTINGS = load_elx2_llm(READER_DIR)

EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
INDEX_PATH_WRITE = './data/one_pdf/one_pdf_rag_index/' #pdfs made by get_vector_store
INDEX_PATH_READ = './data/one_pdf/one_pdf_rag_index/pdfs' #pdfs made by get_vector_store

FILES_PATH = './data/one_pdf/pdf_file'
#for path in [FILES_PATH, INDEX_PATH]:
if not os.path.exists(INDEX_PATH_WRITE):
    os.makedirs(INDEX_PATH_WRITE)



if not os.path.exists(FILES_PATH):
    os.makedirs(FILES_PATH)
else:
    files = glob.glob(os.path.join(FILES_PATH, '*'))
    for f in files:
        os.remove(f)
##################################################################


def answer_with_rag(
    question: str,
    reader_llm: ExLlamaV2StreamingGenerator,
    reader_llm_settings:ExLlamaV2Sampler.Settings,
    embedding_model,
    max_new_tokens,
    knowledge_index,
    use_reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 20, #30,
    num_docs_final: int = 10,
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
    print(f"{Fore.BLUE}=> Retrieving documents...{Style.RESET_ALL}")
    embedding_vector = embedding_model.embed_query(question)
    relevant_docs = knowledge_index.similarity_search_by_vector(embedding_vector, k = num_retrieved_docs)#num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text

    print(f"Len of relevant docs: {len(relevant_docs)}")
    if use_reranker:
        print(f'RELEVANT DOC: {relevant_docs[0]} with type ',type(relevant_docs[0]))
        relevant_docs = RERANKER.rerank(question, relevant_docs, k=num_docs_final)
        print('Done with reranker')
        #relevant_docs = [doc.page_content for doc in relevant_docs] 
        relevant_docs = relevant_docs[:num_docs_final]
    
    RAG_PROMPT_TEMPLATE = """
    Using the information contained in the context,
    give a comprehensive answer to the question.
    Respond only to the question asked, response should be concise and relevant to the question.
    Provide the number of the source document when relevant.
    If the answer cannot be deduced from the context, do not give an answer.</s>
    Context:
    {context}
    ---
    Now here is the question you need to answer.

    Question: {question}

    """
    # Build the final prompt
    context = "\nExtracted documents:\n"
    #type(f"TYPE: {relevant_docs[0]['content']}")
    if use_reranker: #reranker seems to have side effects, changing inputs (will return dicts with 'content' key)
        context += "".join([f"Document {str(i)}:::\n" + doc['content'] for i, doc in enumerate(relevant_docs)])
    else:
        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)
   
    reader_llm.warmup()

    answer = reader_llm.generate_simple(f"<s>[INST] {final_prompt} [/INST]", 
    reader_llm_settings, max_new_tokens, seed = 1234)
    
    inst_idx = answer.find('[/INST]')
    answer =  answer[inst_idx+7:]

    return answer,relevant_docs

def get_10_qas(arxiv_id):
    #get 10 questions
    pass
def upload_file(source_path, destination_dir):
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Copy the file to the destination directory
    shutil.copy(source_path, destination_dir)

    print(f"{Fore.BLUE}File uploaded to {destination_dir}{Style.RESET_ALL}")


def rag_pdf(question,reader_llm, reader_settings):
    embedder,core_embedding_model = get_embedder(EMBED_MODEL)
    vector_store = FAISS.load_local(INDEX_PATH_READ, embedder,allow_dangerous_deserialization=True) 
    embedder = embedder
    #question = input("Enter your question")
    #call get_q_and_a.py, make special directory for this!!!
    answer, _ = answer_with_rag(question, 
                                reader_llm=reader_llm,
                                reader_llm_settings=reader_settings,
                                embedding_model=core_embedding_model,
                                max_new_tokens=1024,
                                knowledge_index=vector_store , 
                                use_reranker = False)#+True)
    return answer

def main():
    print(f"{Fore.BLUE}RAG over arXiv{Style.RESET_ALL}")
    # standard_or_custom = input("""Press 0 to get a standard set of questions (must provide arXiv id for an article posted on or after Dec. 2024) 
    #                            or 1 to enter your own question""")
    standard_or_custom = input(f"""{Fore.GREEN}Press 0 to get a standard set of questions (must provide arXiv id for an article posted on or after Dec. 2023) 
                               or 1 to enter your own question:\n{Style.RESET_ALL}""")
    if standard_or_custom == '0':
        st.write('Standard set of questions')
        arxiv_id = input("Enter the arXiv id of the article")
        #call get_q_and_a.py, make special directory for this!!!
    elif standard_or_custom == '1':
        #print(f'{Fore.BLUE}Upload a PDF file to {FILES_PATH}{Style.RESET_ALL}')
        #st.write('Enter your own question')
        if not os.path.exists(FILES_PATH):
            os.makedirs(FILES_PATH)
        #else:
        files = glob.glob(os.path.join(FILES_PATH, '*'))
        for f in files:
            os.remove(f)
        source_path = input(f"{Fore.GREEN}Provide a path to the PDF file:\n{Style.RESET_ALL}")
        #pdf_path = input()
        upload_file(source_path, FILES_PATH)

        print(f"{Fore.BLUE}Generating vector store and stored at {INDEX_PATH_READ}{Style.RESET_ALL}")
        #print(f'{Fore.BLUE}RUNNING get_vector_store.py')
        subprocess.run(['python', 'scripts/get_vector_store.py', '--pdf_or_txt=' + 'pdf', '--files_path=' + FILES_PATH, 
                        '--chunk_size=' + str(1000), '--chunk_overlap=' + str(200),'--index_dir='+INDEX_PATH_WRITE])
        READER_LLM, READER_LLM_SETTINGS = load_elx2_llm(READER_DIR)
        while True:
            question = input(f"{Fore.GREEN}Enter your question or enter Control+C to exit: {Style.RESET_ALL}")
            #question = input("Enter your own question or enter Control+C to exit:\n")
            if question:
                answer = rag_pdf(question,reader_llm=READER_LLM, reader_settings=READER_LLM_SETTINGS)
                print(answer)
           

if __name__=='__main__':
    # READER_LLM, READER_LLM_SETTINGS = load_elx2_llm(READER_DIR)
    #print('Hello World!')
    main()
    