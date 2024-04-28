import re
import numpy as np
import logging
import feedparser
import os
import shutil
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from exllamav2 import *
from exllamav2.generator import *
from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from langchain.storage import LocalFileStore

def classify_topic_re(example):
    """Classify abstract based on regex pattern"""
    pattern = r'^(?!.*\b(diffusion|3D|computer vision|image|video|resnet|cnn|vit)\b).*\b(lms|attention|language model(s)?|LLMs|context LMs|synthetic data|GPT|RLHF|DPO|KTO|ORPO|.*RNN.*|llama|mamba).*'
    match = re.findall(pattern, example['abstract'], re.IGNORECASE)
    return 'llms' if match else 'other'

def get_title(arxiv_id):
    """
    Fetches the title of a paper from the arXiv API.

    This function constructs a query URL using the provided arXiv ID, fetches the data from the arXiv API, and extracts the title of the paper. If an error occurs, it logs the error and returns an empty string.

    Args:
        arxiv_id (str): The arXiv ID of the paper.

    Returns:
        str: The title of the paper, or an empty string if an error occurs.
    """
    logging.basicConfig(filename='./logs/get_title.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s')
    try:
        url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'

        # Fetch the data
        data = feedparser.parse(url)

        # Get the first entry
        entry = data.entries[0]

        # Get title and abstract
        title = entry.title.replace('\n', '').replace('\r', '').strip()
    
        return title
    except Exception as e:
        print(f"Error: {e}")
        logging.exception(f'Error occurred in getting title for {arxiv_id}: {e}')
        return ''
    
def get_title_and_abstract(arxiv_id):
    # Construct the query URL
    try:
        url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'

        # Fetch the data
        data = feedparser.parse(url)

        # Get the first entry
        entry = data.entries[0]

        # Get title and abstract
        title = entry.title.replace('\n', '').replace('\r', '').strip()
        abstract = entry.summary
        return title, abstract
    except Exception as e:
        print(f"Error: {e}")
        return '',''
    
def move_processed_files(input_dir, processed_dir, filename):
    """
    Moves a processed file from one directory to another.

    This function constructs the full paths to the input and output files and then moves the file from the input directory to the processed directory.

    Args:
        input_dir (str): The directory where the input file is currently located.
        processed_dir (str): The directory where the processed file should be moved to.
        filename (str): The name of the file to be moved.

    Returns:
        None
    """
    input_file = os.path.join(input_dir, filename)
    processed_file = os.path.join(processed_dir, filename)
    shutil.move(input_file, processed_file)


def get_docs_from_txt(files_path:str,text_splitter):
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
    files_path = Path(files_path)
    files = list(files_path.glob('*.txt'))
    if not files: print('Please check the path to the txt files');exit(1)
    print(f'Number of txt files: {len(files)}')
    all_docs = []
    for i in range(len(files)):
        doc = text_splitter.create_documents([files[i].read_text()],metadatas=[{'filename':files[i].name,'title':get_title(files[i].name.split('_')[0])}])
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


def get_docs_from_pdf(files_path:str,text_splitter):
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
    files_path = Path(files_path)
    files = list(files_path.glob('*.pdf'))
    if not files: print('Please check the path to the pdf files');exit(1)
    print(f'Number of txt files: {len(files)}')
    all_docs = [load_pdf_to_string(os.path.expanduser(pdf_path)) for  pdf_path in files]
    docs_processed  = [text_splitter.split_documents([Document(page_content=doc, metadata={'filename':files[idx].name,'title':get_title(files[idx].name.split('_')[0])})]) 
            for idx,doc in enumerate(all_docs)]
    docs_processed = [txt for doc in docs_processed for txt in doc]
    return docs_processed


def load_elx2_llm(model_dir="../MixtralInference/Mixtral-8x7B-instruct-exl2"):
    """
    Loads the ExLlamaV2 language model and prepares it for text generation.

    This function initializes the ExLlamaV2 model with the provided model directory, prepares a cache for the model, and sets up a generator for text generation with the model.

    Args:
        model_dir (str): The directory where the ExLlamaV2 model files are located. Defaults to "/home/mainuser/Desktop/LLMs/MixtralInference/Mixtral-8x7B-instruct-exl2".

    Returns:
        tuple: A tuple containing the ExLlamaV2StreamingGenerator instance (generator) and the ExLlamaV2Sampler.Settings instance (gen_settings).
    """
    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.prepare()

    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, lazy = True)

    print("Loading model...")
    model.load_autosplit(cache)

    tokenizer = ExLlamaV2Tokenizer(config)
    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
    generator.set_stop_conditions([tokenizer.eos_token_id])
    gen_settings = ExLlamaV2Sampler.Settings()

    return generator, gen_settings

def call_llm(
    question: str,
    generator: ExLlamaV2StreamingGenerator,
    settings:ExLlamaV2Sampler.Settings,
    max_new_tokens = 512
) -> str:
    """
    Generate an output from a given question using a specified generator and settings.

    Parameters:
    question (str): The question to generate an output from.
    generator (ExLlamaV2StreamingGenerator): The generator to use for generating the output.
    settings (ExLlamaV2Sampler.Settings): The settings to use for the generator.
    max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.

    Returns:
    Tuple[str, List[LangchainDocument]]: The generated output and a list of LangchainDocument objects.
    """
    max_new_tokens = max_new_tokens

    generator.warmup()
    output = generator.generate_simple(f"<s>[INST] {question} [/INST]", settings, max_new_tokens, seed = 1234)
    return output

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
