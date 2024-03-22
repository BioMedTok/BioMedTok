# SubwordAnalysis.py

import json
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer, CamembertTokenizer, RobertaTokenizerFast

models = [
    "BioMedTok/BPE-HF-NACHOS-FR",
    "BioMedTok/BPE-HF-NACHOS-FR-Morphemes",

    "BioMedTok/BPE-HF-PubMed-FR",
    "BioMedTok/BPE-HF-PubMed-FR-Morphemes",
    
    "BioMedTok/BPE-HF-CC100-FR",
    "BioMedTok/BPE-HF-CC100-FR-Morphemes",
    
    "BioMedTok/BPE-HF-Wikipedia-FR",
    "BioMedTok/BPE-HF-Wikipedia-FR-Morphemes",
    
    "BioMedTok/SentencePieceBPE-NACHOS-FR",
    "BioMedTok/SentencePieceBPE-NACHOS-FR-Morphemes",
    
    "BioMedTok/SentencePieceBPE-PubMed-FR",
    "BioMedTok/SentencePieceBPE-PubMed-FR-Morphemes",
    
    "BioMedTok/SentencePieceBPE-CC100-FR",
    "BioMedTok/SentencePieceBPE-CC100-FR-Morphemes",
    
    "BioMedTok/SentencePieceBPE-Wikipedia-FR",
    "BioMedTok/SentencePieceBPE-Wikipedia-FR-Morphemes",
]

tasks = [
    {"model": "BioMedTok/DiaMED", "subset": None, "dataset": None, "data_path": "./recipes/diamed/data/"},
]

matrix_avg_tokens_per_word = {f"{t['model']}-{t['subset']}": {m: [] for m in models} for t in tasks}

for m in models:

    print(f">> {m}")

    if "sentencepiece" in m.lower():
        tokenizer = CamembertTokenizer.from_pretrained(m)
    elif "bpe-hf" in m.lower():
        tokenizer = RobertaTokenizerFast.from_pretrained(m, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(m)
    
    ds = load_dataset("BioMedTok/DiaMED")["test"]

    text = ds[0]['clinical_case']
    print(text)
    exit(0)
    subwords = tokenizer.tokenize(text)
    print(m, " - ", subwords)

    # tokens = e['tokens']
    # output = tokenizer(list(tokens), is_split_into_words=True)["input_ids"]
