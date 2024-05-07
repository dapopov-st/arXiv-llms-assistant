# arXiv LLMs Assistant

![Project Structure](./assets/arxiv-assistant.png)
Image by author, made with [whimsical](https://whimsical.com)
## In Progress

### The goal is twofold:
- Assist in studying the LLM domain by comparing out-of-repository papers with papers in a personal repo (in Zotero), then recommending new papers to read along with providing a list of question/answer pairs for each recommended paper
- Run RAG over new papers, helping with generating questions to gain a deep understanding, for example.  For this purpose, RAG is evaluated with different configurations and the best evaluated configuration is selected.

### Evaluation
- The original starting point was a notebook on [RAG Evaluation](https://huggingface.co/learn/cookbook/en/rag_evaluation#evaluating-rag-performance) by HuggingFace
- There are LLM multiple models involved in evaluation pipeline, and to fit it all on two 3090 GPUs, the code was refactored into scripts (so that each subprocess would release the GPU memory once done)
- In addition, to conserve memory and increase speed, 6bpw exl2 quantizations were used for Mistral and Mixtral due to their excellent [perplexity scores](https://huggingface.co/turboderp/Mixtral-8x7B-instruct-exl2). 
- Finally, in the spirit of keeping all parts of the projects open source, Prometheus Eval was used (8bpw exl2) as an alternative to GPT-4 evaluation
- A rougth draft of evaluation results graph is below.  Retrieving from PDFs with chunk size of 1000 and 200 overlap yielded the best results, beating the best setting for retrieving from HTML markups converted to plain text (chunk size of 2000 with 200 overlap), although the gap is
- Result comparison forthcoming
 ![Evaluation Scores](./assets/eval_scores.png)
### Other findings and observations
- It's best to preprocess the data by only loading the articles up to the References section as References were first to be retrieved otherwise, which is not useful
- Forthcoming

### Installation and usage instructions
- Forthcoming: Selenium, Mixtral exl2

### Potential future directions
- Currently, papers are selected from LLM subset of AK's recommendations for the past week using 80/20 rule.  This makes the selection simple and robust, but an approach such as nearest neighbors or continuous learning, for example, could be more rigorous
- The topic can be changed to a non-LLM topic or broadened by modifying classify_topic_re in scripts/utils.py
- Currently, the implementation is tied to looking for existing papers in Zotero.  This can be easily relaxed and a list of arxiv ids can be used as a starting point instead (replace scripts/get_arxiv_nums.py get_arxiv_nums function with one that takes a list of arxiv ids and writes to directory)
- Mixtral evaluation with Reranker prooved to be a challenge due to device management.  Since desired performance was already achieved, this is left to future work

### Command line utility 
To meet the second goal, the best setting out of the evaluated results was chosen and used to build a command line utility. The user would specify a path to the pdf file and then either specify 0 to get a standard set of questions or 1 to ask custom questions as shown below:
![Cmd utility demo](./assets/cmdline_demo.png)

   
