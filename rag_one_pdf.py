
import subprocess
import json
import base64
from argparse import ArgumentParser
from tqdm.auto import tqdm
import sys, os
import glob
from scripts.utils import load_elx2_llm, get_embedder

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
# print(f'SESSION STATE: {st.session_state}')
# if 'model_loaded' not in st.session_state:
#     #READER_LLM, READER_LLM_SETTINGS = load_elx2_llm(READER_DIR)
#     st.session_state.READER_LLM, st.session_state.READER_LLM_SETTINGS = load_elx2_llm(READER_DIR)
#     st.session_state.model_loaded = True
EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
INDEX_PATH_WRITE = './data/one_pdf/one_pdf_rag_index/' #pdfs made by get_vector_store
INDEX_PATH_READ = './data/one_pdf/one_pdf_rag_index/pdfs' #pdfs made by get_vector_store

FILES_PATH = './data/one_pdf/pdf_file'
#for path in [FILES_PATH, INDEX_PATH]:
if not os.path.exists(INDEX_PATH_WRITE):
    os.makedirs(INDEX_PATH_WRITE)


if 'files_removed' not in st.session_state:
    if not os.path.exists(FILES_PATH):
        os.makedirs(FILES_PATH)
    else:
        files = glob.glob(os.path.join(FILES_PATH, '*'))
        for f in files:
            os.remove(f)
    st.session_state.files_removed = True
    
st.title(f"arXiv LLMs Assistant")
##################################################################


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-color: rgba(255, 255, 255, 0.025);
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

#set_background("./assets/streamlit-background.png")


##################################################################

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
    #relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text


    if use_reranker:
        relevant_docs = RERANKER.rerank(question, relevant_docs, k=num_docs_final)
    relevant_docs = [doc.page_content for doc in relevant_docs] 


    relevant_docs = relevant_docs[:num_retrieved_docs]
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
    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)
   
    reader_llm.warmup()

    answer = reader_llm.generate_simple(f"<s>[INST] {final_prompt} [/INST]", 
    reader_llm_settings, max_new_tokens, seed = 1234)



    return answer,relevant_docs



def get_10_qas(arxiv_id):
    #get 10 questions
    pass
def upload_file():
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    if uploaded_file:
        with open(os.path.join(FILES_PATH, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
            return True
        #rag_pdf(uploaded_file.name)
    return False

def rag_pdf(question):
    if 'embedder' not in st.session_state:
        st.session_state.embedder,st.session_state.core_embedding_model = get_embedder(EMBED_MODEL)
        st.session_state.vector_store = FAISS.load_local(INDEX_PATH_READ, st.session_state.embedder,allow_dangerous_deserialization=True) 
    #question = input("Enter your question")
    #call get_q_and_a.py, make special directory for this!!!
    answer, _ = answer_with_rag(question, 
                                reader_llm=st.session_state.READER_LLM,
                                reader_llm_settings=st.session_state.READER_LLM_SETTINGS,
                                embedding_model=st.session_state.core_embedding_model,
                                max_new_tokens=1024,
                                knowledge_index=st.session_state.vector_store , 
                                use_reranker = False) #True)
    return answer

def main():
    print(f'SESSION STATE in main: {st.session_state}')
    if 'background_set' not in st.session_state:
        set_background("./assets/streamlit-background.jpg")
        st.session_state.background_set = True

    # standard_or_custom = input("""Press 0 to get a standard set of questions (must provide arXiv id for an article posted on or after Dec. 2024) 
    #                            or 1 to enter your own question""")
    standard_or_custom = st.text_input("""Press 0 to get a standard set of questions (must provide arXiv id for an article posted on or after Dec. 2023) 
                               or 1 to enter your own question""")
    if standard_or_custom == '0':
        st.write('Standard set of questions')
        arxiv_id = input("Enter the arXiv id of the article")
        #call get_q_and_a.py, make special directory for this!!!
    elif standard_or_custom == '1':
        st.markdown("### Upload a PDF file")
        file_uploaded = upload_file()
        st.write('Enter your own question')
        
        if file_uploaded:
            print("Generating vector store and stored at " + INDEX_PATH_READ)
            st.markdown('RUNNING get_vector_store.py')
            subprocess.run(['python', 'scripts/get_vector_store.py', '--pdf_or_txt=' + 'pdf', '--files_path=' + FILES_PATH, 
                            '--chunk_size=' + str(1000), '--chunk_overlap=' + str(200),'--index_dir='+INDEX_PATH_WRITE])
            st.session_state.vector_store_generated = True
            st.session_state.READER_LLM, st.session_state.READER_LLM_SETTINGS = load_elx2_llm(READER_DIR)
            st.session_state.model_loaded = True
            while True:
                question = st.text_input("Enter your own question or enter Control+C to exit")
                if question:
                    answer = rag_pdf(question)#,reader_llm=READER_LLM, reader_settings=READER_LLM_SETTINGS)
                    inst_idx = answer.find('[INST]')
                    st.markdown(answer[inst_idx+6:])
        #     question = st.text_input("Enter your own question")
        #     answer = rag_pdf(question)
        #     print(answer)
        #     st.markdown(answer)

        if file_uploaded and 'vector_store_generated' not in st.session_state:
            print("Generating vector store and stored at " + INDEX_PATH_READ)
            st.session_state.vector_store_generated = True
            st.markdown('RUNNING get_vector_store.py')
            subprocess.run(['python', 'scripts/get_vector_store.py', '--pdf_or_txt=' + 'pdf', '--files_path=' + FILES_PATH, 
                            '--chunk_size=' + str(1000), '--chunk_overlap=' + str(200),'--index_dir='+INDEX_PATH_WRITE])
            # st.session_state.vector_store_generated = True

        # if 'vector_store_generated' in st.session_state:
        #     question = st.text_input("Enter your own question")
        #     if question and 'answer_written' not in st.session_state:
        #         answer = rag_pdf(question)
        #         print(answer)
        #         st.markdown(answer)
        #         st.session_state.answer_written = True


        # # UNCOMMENT
        # if 'vector_store_generated' in st.session_state:
        #     question = st.text_input("Enter your own question")
        #     if question:
        #         if 'answer' not in st.session_state:
        #             st.session_state.answer = rag_pdf(question)
        #             print(st.session_state.answer)
        #         st.markdown(st.session_state.answer)

if __name__=='__main__':
    print(f'SESSION STATE: {st.session_state}')
    # if 'model_loaded' not in st.session_state:
    #     #READER_LLM, READER_LLM_SETTINGS = load_elx2_llm(READER_DIR)
    #     st.session_state.READER_LLM, st.session_state.READER_LLM_SETTINGS = load_elx2_llm(READER_DIR)
    #     st.session_state.model_loaded = True
    main()

#TODO: WRAP UP ^ (make dir for 10 qs when uploaded)
#THEN make appliction work for similar articles to list of available papers by arxiv_ids
#Make directories for all data, perhaps other necessary