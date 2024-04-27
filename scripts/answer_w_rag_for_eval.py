from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
store = LocalFileStore("./cache/")

#embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
embed_model_id = 'mixedbread-ai/mxbai-embed-large-v1'
core_embeddings_model = HuggingFaceEmbeddings(
    model_name=embed_model_id
)
embedder = CacheBackedEmbeddings.from_bytes_store(
    core_embeddings_model, store, namespace=embed_model_id
)

vector_store = FAISS.load_local('../data/rag_index_dir', embedder,allow_dangerous_deserialization=True)
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
from exllamav2 import *
from exllamav2.generator import *
import sys, torch


reader_config = ExLlamaV2Config()
reader_config.model_dir = "/home/mainuser/Desktop/LLMs/ZephyrInference"
#reader_config.model_dir = '/home/mainuser/Desktop/LLMs/Mixtral4bit'
reader_config.prepare()

reader_model = ExLlamaV2(reader_config)
cache = ExLlamaV2Cache(reader_model, lazy = True)

print("Loading model...")
reader_model.load_autosplit(cache)

reader_tokenizer = ExLlamaV2Tokenizer(reader_config)
reader_llm = ExLlamaV2StreamingGenerator(reader_model, cache, reader_tokenizer)
#reader_llm.set_stop_conditions([reader_tokenizer.eos_token_id])
reader_settings = ExLlamaV2Sampler.Settings()
reader_settings.temperature = 0.85
reader_settings.top_k = 30
reader_settings.top_p = 0.8
reader_settings.token_repetition_penalty = 1.03
from ragatouille import RAGPretrainedModel
from typing import Optional, List, Tuple
RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
from langchain.docstore.document import Document as LangchainDocument
def answer_with_rag(
    question: str,
    generator: ExLlamaV2StreamingGenerator,
   # tokenizer: ExLlamaV2Tokenizer,
    settings:ExLlamaV2Sampler.Settings,
    max_new_tokens = 512,
    knowledge_index: FAISS = vector_store,
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 10, #30,
    #num_docs_final: int = 5,
) -> Tuple[str, List[LangchainDocument]]:
    # Gather documents with retriever
    print("=> Retrieving documents...")
    embedding_vector = core_embeddings_model.embed_query(question)
    relevant_docs = knowledge_index.similarity_search_by_vector(embedding_vector, k = num_retrieved_docs)#num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text

    # Optionally rerank results
    # if reranker:
    #     print("=> Reranking documents...")
    #     relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
    #     #print(f"Type is : {type(relevant_docs[0])}")
    #     print(dir(relevant_docs[0]))
    #     relevant_docs = [doc['page_content'] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_retrieved_docs]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

   
    generator.warmup()
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    answer = generator.generate_simple(final_prompt, 
    settings, max_new_tokens, seed = 1234)
    # print(answer)
    return answer,relevant_docs


answer, relevant_docs = answer_with_rag(question="What is the difference between RAG and self-RAG?", generator=reader_llm,settings=reader_settings,max_new_tokens=512)
from ragatouille import RAGPretrainedModel
from typing import Optional, List, Tuple
RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
from langchain.docstore.document import Document as LangchainDocument
def answer_with_rag(
    question: str,
    generator: ExLlamaV2StreamingGenerator,
   # tokenizer: ExLlamaV2Tokenizer,
    settings:ExLlamaV2Sampler.Settings,
    max_new_tokens = 512,
    knowledge_index: FAISS = vector_store,
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 10, #30,
    num_docs_final: int = 5,
) -> Tuple[str, List[LangchainDocument]]:
    # Gather documents with retriever
    print("=> Retrieving documents...")
    embedding_vector = core_embeddings_model.embed_query(question)
    relevant_docs = knowledge_index.similarity_search_by_vector(embedding_vector, k = num_retrieved_docs)#num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text


    if reranker:
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]


    relevant_docs = relevant_docs[:num_retrieved_docs]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

   
    generator.warmup()
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    answer = generator.generate_simple(final_prompt, 
    settings, max_new_tokens, seed = 1234)
    return answer,relevant_docs


answer, relevant_docs = answer_with_rag(question="What is the difference between RAG and self-RAG?", generator=reader_llm,settings=reader_settings,max_new_tokens=512,reranker = RERANKER)
