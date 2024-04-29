"""
Note: This script can be used to classify the topic of an arXiv abstract into one of the following categories: LLMs, diffusion, computer vision, multimodal.
I ended up using regex approach for this (in utils), but if this script is imported, it will configure the LLM model and classify the topic based on the abstract.
Note that the model will need 8-9GB of GPU memory to run.
"""

from exllamav2 import *
from exllamav2.generator import *
import re


generator_config = ExLlamaV2Config()
generator_config.model_dir = "/home/mainuser/Desktop/LLMs/MiStralInference" #Change path to the model directory
generator_config.prepare()

generator_model = ExLlamaV2(generator_config)
cache = ExLlamaV2Cache(generator_model, lazy = True)

print("Loading model...")
generator_model.load_autosplit(cache)

generator_tokenizer = ExLlamaV2Tokenizer(generator_config)
generator_llm = ExLlamaV2StreamingGenerator(generator_model, cache, generator_tokenizer)
generator_llm.set_stop_conditions([generator_tokenizer.eos_token_id])
generator_settings = ExLlamaV2Sampler.Settings()
generator_settings.temperature = 0.85
generator_settings.top_k = 50
generator_settings.top_p = 0.8
generator_settings.token_repetition_penalty = 1.01

def call_llm(
    question: str,
    generator: ExLlamaV2StreamingGenerator,
    settings:ExLlamaV2Sampler.Settings,
    max_new_tokens = 512
    ):

    max_new_tokens = max_new_tokens

    generator.warmup()
    output = generator.generate_simple(f"<s>[INST] {question} [/INST]", settings, max_new_tokens, seed = 1234)
    return output

def classify_topic(abstract):
    topic_classification_prompt = f"""
    Your task is to take an arXiv abstract and classify it into one of the following categories: LLMs, diffusion, computer vision, multimodal.
    If you see LLMs or language models in the abstract, classify as LLMs. 
    If you see multimodal in abstract, classify as multimodal.  If you see diffusion in the abstract, classify as diffusion. If you see video, vision, or visual in the abstract, classify as computer vision. 
    Now here is the abstract: {abstract}
    Think step-by-step about your answer and check your answer before returning it and provide a brief explanation.

    Category:
    """
    ans = call_llm(question=topic_classification_prompt, generator=generator_llm,settings=generator_settings,max_new_tokens=24)[len(topic_classification_prompt):]
    pattern = r'(LLMs|diffusion|computer vision|multimodal)'
    match = re.search(pattern, ans,re.IGNORECASE)
    response =  match.group(1) if match else 'other'
    return response.lower()