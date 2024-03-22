import torch
from tqdm import tqdm

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer, CamembertTokenizer, RobertaTokenizerFast, CamembertForMaskedLM

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

print("Start loading!")
f_in = open("/users/ylabrak/TokenizationDrBERT/nachos_subset_perplexity_500MB.txt","r")
texts = [t for t in f_in.read().split("\n") if len(t) >= 0]
f_in.close()
print("Finished loading!")

device = "cuda"

for model_name in models:

    print(model_name)

    if "sentencepiece" in model_name.lower():
        tokenizer = CamembertTokenizer.from_pretrained(model_name)
    elif "bpe_tokenizers" in model_name.lower() and "-roberta" in model_name.lower():
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
    elif "bpe_tokenizers" in model_name.lower():
        m_name = model_name.split("/")[-1]
        f_vocab = f"{model_name}/{m_name}-vocab.json"
        f_merges = f"{model_name}/{m_name}-merges.txt"
        tokenizer = ByteLevelBPETokenizer(f_vocab, f_merges)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = CamembertForMaskedLM.from_pretrained(model_name).to(device)
    # tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(ppl)