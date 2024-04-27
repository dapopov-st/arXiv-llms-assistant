"""
get_vector_store.py

This script processes a set of text or PDF files, creating a list of Document objects from their content, and 
generates a FAISS vector store from these documents.

The script uses several classes and functions from the langchain library, including RecursiveCharacterTextSplitter, 
Document, CacheBackedEmbeddings, HuggingFaceEmbeddings, FAISS, and LocalFileStore. It also uses the PyPDF2 library to 
read PDF files.

The script defines two functions, get_docs_from_txt and get_docs_from_pdf, which read a list of text or PDF files 
respectively and create a list of Document objects from their content.

Usage:
    python get_vector_store.py --pdf_or_txt='pdf' --files_path='./data/pdfs_ws_mrkp_test/pdfs/'

Arguments:
    --pdf_or_txt: Specifies the type of files to process. It should be either 'pdf' for PDF files or 'txt' for text files. The default value is 'txt'.

    --embed_model_id: Specifies the ID of the HuggingFace model to use for embeddings. The default value is 'mixedbread-ai/mxbai-embed-large-v1'.

    --index_dir: Specifies the path to the directory where the FAISS index should be stored. The default value is './data/rag_index_dir'.

    --files_path: Specifies the path to the directory containing the files to process. The default value is './data/pdfs_ws_mrkp_test/mrkps/'.

    --chunk_size: Specifies the size of the chunks to split the text into when creating Document objects. The size is measured in number of characters. The default value is 2000.

    --chunk_overlap: Specifies the number of characters that should overlap between consecutive chunks when splitting the text into chunks. The default value is 200.

Note: This script relies on a global 'args' object for its parameters. The 'args' object should have 'chunk_size', 
'chunk_overlap', 'embed_model_id', 'files_path', and 'pdf_or_txt' attributes.
"""

from tqdm.auto import tqdm
from argparse import ArgumentParser
import pandas as pd
pd.set_option("display.max_colwidth", None)

from pathlib import Path
import sys, os
cwd = os.getcwd()
print(cwd)
sys.path.append(os.path.join(cwd, 'scripts'))
import utils

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore

from PyPDF2 import PdfReader

def get_docs_from_txt(files,text_splitter):
    """
    This function reads a list of text files and creates a list of Document objects from their content.

    Parameters:
    files (list): A list of text file objects to read. Each object should have a 'read_text' method to read the file 
                  content and a 'name' attribute for the file name.
    text_splitter (object): An object that can split text into chunks and create Document objects from these chunks. 
                            This object should have a 'create_documents' method that takes a list of text strings and 
                            a list of metadata dictionaries, and returns a list of Document objects.

    Returns:
    list: A list of Document objects created from the text files. Each Document object has a 'metadata' attribute 
          containing a dictionary with 'filename' and 'title' keys.

    """
    all_docs = []
    for i in range(len(files)):
        doc = text_splitter.create_documents([files[i].read_text()],metadatas=[{'filename':files[i].name,'title':utils.get_title(files[i].name.split('_')[0])}])
        all_docs.extend(doc)
    return all_docs


def load_pdf_to_string(pdf_path):
    """
    This function reads a PDF file and extracts its text content up to the 'REFERENCES' section.

    Parameters:
    pdf_path (str): The path to the PDF file to read.

    Returns:
    str: A string containing the text content of the PDF file up to (but not including) the 'REFERENCES' section.

    """
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            references_index= page_text.upper().find('\nREFERENCES\n')
            if references_index != -1:
              page_text = page_text[:references_index]
              text += page_text
              return text
            text += page_text
    return text


def get_docs_from_pdf(files,text_splitter):
    """
    This function reads a list of PDF files and creates a list of Document objects from their content.

    Parameters:
    files (list): A list of PDF file paths to read. Each path should be a string.
    text_splitter (object): An object that can split text into chunks and create Document objects from these chunks. 
                            This object should have a 'split_documents' method that takes a list of Document objects 
                            and returns a list of split Document objects.

    Returns:
    list: A list of Document objects created from the PDF files. Each Document object has a 'metadata' attribute 
          containing a dictionary with 'filename' and 'title' keys.
    """
    all_docs = [load_pdf_to_string(os.path.expanduser(pdf_path)) for  pdf_path in files]
    docs_processed  = [text_splitter.split_documents([Document(page_content=doc, metadata={'filename':files[idx].name,'title':utils.get_title(files[idx].name.split('_')[0])})]) 
            for idx,doc in enumerate(all_docs)]
    docs_processed = [txt for doc in docs_processed for txt in doc]
    return docs_processed



def get_embedder(embed_model_id='mixedbread-ai/mxbai-embed-large-v1'):
    """
    This function creates an embedder object that can be used to transform text into vector representations.

    Parameters:
    embed_model_id (str, optional): The ID of the HuggingFace model to use for embeddings. 
                                    Defaults to 'mixedbread-ai/mxbai-embed-large-v1'.

    Returns:
    tuple: A tuple containing two elements:
        - embedder (CacheBackedEmbeddings object): An embedder object that can be used to transform text into vector 
                                                   representations. This object is backed by a cache, so if the same 
                                                   text is embedded multiple times, the cached result will be used 
                                                   instead of recomputing the embedding.
        - core_embeddings_model (HuggingFaceEmbeddings object): The underlying HuggingFace model used for embeddings.
    """
    store = LocalFileStore("./cache/")

    embed_model_id = embed_model_id
    core_embeddings_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={"trust_remote_code":True}
    )
    embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model, store, namespace=embed_model_id

    )
    return embedder,core_embeddings_model



def get_vector_store(docs, embedder):
    """
    This function generates a FAISS vector store from a list of documents.

    Parameters:
    docs (list): A list of Document objects to be embedded. Each Document object should have a 'metadata' attribute.
    embedder (object): An object that can embed the documents, i.e., transform them into a vector representation. 
                       This object should have a method 'from_documents' that takes a list of Document objects and 
                       returns a FAISS vector store.

    Returns:
    vector_store (object): A FAISS vector store containing the vector representations of the documents. The vector 
                           store has a 'merge_from' method that can merge another vector store into it.
    """
    for i, doc in enumerate(docs):
        doc.metadata['vs_index'] = i
        if i == 0:
            vector_store = FAISS.from_documents([doc], embedder)
        else:
            vector_store_i = FAISS.from_documents([doc], embedder)
            vector_store.merge_from(vector_store_i)
    return vector_store



def generate_vs():
    """
    This function processes a set of text or PDF files, creating a list of Document objects from their content.

    The function first creates a RecursiveCharacterTextSplitter object and an embedder object. It then checks the 
    'pdf_or_txt' argument to determine whether to process text or PDF files. It reads the files from the specified 
    path, processes them into Document objects using the appropriate function (get_docs_from_txt or get_docs_from_pdf), 
    and prints the number of files and documents processed.

    If 'pdf_or_txt' is not 'txt' or 'pdf', the function prints an error message and exits.

    Note: This function relies on global 'args' object for its parameters. The 'args' object should have 'chunk_size', 
    'chunk_overlap', 'embed_model_id', 'files_path', and 'pdf_or_txt' attributes.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap = args.chunk_overlap,
        length_function=len,
    )

    embedder,_ = get_embedder(args.embed_model_id)
    FILES_PATH = Path(args.files_path)
    
    if args.pdf_or_txt == 'txt':
        FILES = list(FILES_PATH.glob('*.txt'))
        if not FILES: print('Please check the path to the txt files');exit(1)
        print(f'Number of txt files: {len(FILES)}')
        docs_processed = get_docs_from_txt(FILES,text_splitter=text_splitter)
        print(f'Number of documents: {len(docs_processed)}')
        path_sub = 'txts'
    elif args.pdf_or_txt == 'pdf':
        FILES = list(FILES_PATH.glob('*.pdf'))
        if not FILES: print('Please check the path to the pdf files');exit(1)
        print(f'Number of pdf files: {len(FILES)}')
        docs_processed = get_docs_from_pdf(FILES,text_splitter=text_splitter)
        print(f'Number of documents: {len(docs_processed)}')
        path_sub = 'pdfs'
    else:
        print('Please specify pdf or txt')
        exit(1)

    
    vector_store = get_vector_store(docs_processed, embedder)
    index_path = os.path.join(args.index_dir,path_sub)
    vector_store.save_local(index_path)
    print(f'Index saved to {index_path}')

parser = ArgumentParser()
parser.add_argument('--pdf_or_txt', type=str, default='txt')
parser.add_argument('--embed_model_id', type=str, default='mixedbread-ai/mxbai-embed-large-v1')
parser.add_argument('--index_dir', type=str, default='./data/rag_index_dir')
parser.add_argument('--files_path', type=str, default='./data/pdfs_ws_mrkp_test/mrkps/')
parser.add_argument('--chunk_size', type=int, default=2000)
parser.add_argument('--chunk_overlap', type=int, default=200)
args = parser.parse_args()

if __name__=='__main__':
    generate_vs()



