# arXiv LLMs Assistant

![Project Structure](./assets/arxiv-assistant2.png)
Image by author, made with [whimsical](https://whimsical.com)
## In Progress

### The goal is twofold:
    - Assist in studying the LLM domain by comparing out-of-repository papers with papers in a personal repo (in Zotero), then recommending new papers to read along with providing a list of question/answer pairs for each recommended paper
    - Run RAG over new papers, helping with generating questions to gain a deep understanding, for example

DONE:

    - Preprocess the data by only loading the articles up to the References section as References were first to be retrieved otherwise, which is not useful
    - Get Mixtral-8x-7b to work with RAG. Answer quality does not go down if Mixtral has 'seen' the paper and Mixtral is able to answer the question due to additional context from RAG if it has.  The response is returned in real time with exl2-quantized 6bit version.
  

   
