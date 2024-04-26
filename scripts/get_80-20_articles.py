"""
To be run as a script; figure out how to add embeds if needed, but making new may be fast enough with FAISS
"""

from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.document import Document
import numpy as np
import os
import argparse
from functools import partial
from utils import classify_topic_re
import pandas as pd  

def get_embedder(embed_model_id):
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

def get_title_abstract_abbrev(directory,filename=None):
    path = os.path.join(directory, filename)
    with open(path, 'r') as file:
        lines = file.readlines()
        title = lines[0].strip()
        abstract = "".join(lines[1:]).replace('\n', ' ')
    return title, abstract, filename.replace('.txt','')

def get_docs(directory):
    existing_files = set(os.listdir(directory))
    docs = [get_title_abstract_abbrev(directory=directory,filename=file) for file in existing_files]
    docs = [Document(page_content=doc[1],metadata={'title':doc[0], 'abbrev':doc[2]}) for doc in docs]
    return docs

#TODO: Perhaps refactor this later to write vector store to a directoy and only add new vectors to it if not already there
#So far, FAISS is blazing fast compared to model load above, so not prematurely optimizing
def get_vector_store(docs, embedder):
    for i, doc in enumerate(docs):
        doc.metadata['vs_index'] = i
        if i == 0:
            vector_store = FAISS.from_documents([doc], embedder)
        else:
            vector_store_i = FAISS.from_documents([doc], embedder)
            vector_store.merge_from(vector_store_i)
    return vector_store


def get_embedding_distances(query,embed_model, vector_store,docs,k=10):
    #core_embeddings_model
    embedding_vector = embed_model.embed_query(query)
    docs = vector_store.similarity_search_by_vector(embedding_vector, k = k)
    vs_indices = [doc.metadata['vs_index'] for doc in docs]
  
    similar_embedding_vectors = np.array([vector_store.index.reconstruct_n(index_id, 1)[0] for index_id in vs_indices])

    distances = np.linalg.norm(similar_embedding_vectors-np.array(embedding_vector), axis=1)
    average_distance = np.mean(distances)
    return distances, average_distance

def get_80_20(df):
    eighty=df.iloc[:4,:]
    twenty  = df[(df['avg_emb_dist']==df.avg_emb_dist.max())]
    return pd.concat([eighty,twenty],axis=0)

def write_8020_to_dir(df):
    if not os.path.exists('./data/abstracts_new/'):
        os.makedirs('./data/abstracts_new/')
    for _,row in df.iterrows():
        with open(f'./data/abstracts_new/{row['arxiv_abbrev']}.txt','w') as f:
            f.write(row['title']+'\n')
            f.write(row['abstract'])

parser = argparse.ArgumentParser(description='Make embeddings for abstracts')
parser.add_argument('--directory_read_txts', type=str, default='./data/abstracts',help='Directory of txt abstracts for papers read')
parser.add_argument('--embed_model_id', type=str, default = 'mixedbread-ai/mxbai-embed-large-v1',help='Huggingface id of the embedder model')
parser.add_argument('--directory_unread_csv', type=str, default = './data/ak_articles_up_to/articles_up_to_2024-04-16.csv',help='Path to new articles csv file')
args = parser.parse_args() 

if __name__=='__main__':

    directory_read_txts = args.directory_read_txts
    embed_model_id = args.embed_model_id
    directory_unread_csv = args.directory_unread_csv

    embedder,core_embeddings_model = get_embedder(embed_model_id)

    docs = get_docs(directory_read_txts)
    vector_store = get_vector_store(docs, embedder) 

    get_dists = partial(get_embedding_distances,embed_model=core_embeddings_model,vector_store=vector_store,docs=docs,k=10)

    new_articles=pd.read_csv(directory_unread_csv)
    new_articles['topic']=new_articles.apply(classify_topic_re,axis=1)

    llm_articles = new_articles[new_articles['topic']=='llms'].copy()
    llm_articles.loc[:, 'emb_dists'], llm_articles.loc[:, 'avg_emb_dist'] = zip(*llm_articles['abstract'].apply(lambda x: get_dists(x)))


    eighty_twenty= get_80_20(df=llm_articles)
    write_8020_to_dir(df=eighty_twenty)