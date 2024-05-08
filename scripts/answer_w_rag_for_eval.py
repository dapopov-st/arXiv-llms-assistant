import argparse
import sys, os
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'scripts'))

import utils


from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
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
    reader: ExLlamaV2StreamingGenerator,
    settings:ExLlamaV2Sampler.Settings,
    embedding_model,
    max_new_tokens,
    knowledge_index,
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 10, #30,
    num_docs_final: int = 5,
) -> Tuple[str, List[LangchainDocument]]:
    print("=> Retrieving documents...")
    embedding_vector = embedding_model.embed_query(question)
    relevant_docs = knowledge_index.similarity_search_by_vector(embedding_vector, k = num_retrieved_docs)#num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text


    if reranker:
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]


    relevant_docs = relevant_docs[:num_retrieved_docs]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

   
    reader.warmup()
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    answer = reader.generate_simple(final_prompt, 
    settings, max_new_tokens, seed = 1234)
    return answer,relevant_docs


def main():
    reader_llm, reader_settings = utils.load_elx2_llm(args.readerllm_dir)
    embedder,core_embeddings_model = utils.get_embedder(args.embed_model_id)
    vector_store = FAISS.load_local(args.vs_dir, embedder,allow_dangerous_deserialization=True)
    answer_with_rag(embedding_model=core_embeddings_model,
                    knowledge_index=vector_store,)
    print(answer)
    print(relevant_docs)
answer, relevant_docs = answer_with_rag(question="What is the difference between RAG and self-RAG?", generator=reader_llm,settings=reader_settings,max_new_tokens=512,reranker = RERANKER)
parser = argparse.ArgumentParser()
parser.add_argument('--readerllm_dir', type=str, default='./data/llm')
parser.add_argument('--vs_dir', type=str, default='./data/rag_index_dir')
args = parser.parse_args()
if __name__ == '__main  __':
    main()