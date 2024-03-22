import json
from transformers import AutoTokenizer, AutoModelForMaskedLM

f_in = open("./morpheme_fr_struct.txt","r")
morphemes = [e.lower() for e in f_in.read().split("\n") if len(e) > 0]
# morphemes = [e.lower() for e in f_in.read().split("\n") if len(e) > 3]
f_in.close()

models = [

    "BioMedTok/BPE-HF-NACHOS-FR",
    "BioMedTok/BPE-HF-PubMed-FR",
    "BioMedTok/BPE-HF-CC100-FR",
    "BioMedTok/BPE-HF-Wikipedia-FR",

    "BioMedTok/SentencePieceBPE-NACHOS-FR",
    "BioMedTok/SentencePieceBPE-PubMed-FR",
    "BioMedTok/SentencePieceBPE-CC100-FR",
    "BioMedTok/SentencePieceBPE-Wikipedia-FR",

]

all_res = {}

def MOD(text):
    text = text.replace("▁","").replace("Ġ","").replace("Ã©","é")
    text = text.replace("Ã¯","ï").replace("Ã¨","è").replace("Ã","à")
    text = text.replace("Ã´","ô").replace("Ã§","ç").replace("Ãª","ê")
    text = text.replace("Ã¹","ù").replace("Å","œ").replace("Ã«","ë")
    text = text.replace("Ã¦","æ").replace("Ã¼","ü").replace("Ã¢","â")
    text = text.replace("â¬","€").replace("Â©","©").replace("Â¤","¤")
    text = text.replace("à®","î").replace("à¢","â").replace("à´","ô")
    text = text.replace("à»","û").replace("à§","ç").replace("à«","ë")
    text = text.replace("à¼","ü")
    return text

for MODEL_NAME in models:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab = [MOD(k) for k in tokenizer.vocab.keys() if len(MOD(k)) > 0]
    # print(vocab)
    # print([voc for voc in vocab if not voc.isalpha()])

    # _r = [v in morphemes for v in vocab]
    # _m = [m in vocab for m in morphemes]
    _m = [m in vocab for m in [mm for mm in morphemes if 1 <= len(mm) and len(mm) <= 10]]

    all_res[MODEL_NAME] = {
        # "percent_of_vocab_in_morphemes": (sum(_r) / len(_r)) * 100,
        "percent_of_morphemes_in_vocab": (sum(_m) / len(_m)) * 100,
    }

    # print("percent_of_vocab_in_morphemes")
    # print(f"Inside: {sum(_r)}")
    # print(f"Total vocab size: {len(_r)}")
    # print("#"*50)

    # print("percent_of_morphemes_in_vocab")
    # print(f"Inside: {sum(_m)}")
    # print(f"Total vocab size: {len(_m)}")
    
    # print(all_res)

    # print(f"{MODEL_NAME}")
    print(all_res[MODEL_NAME]["percent_of_morphemes_in_vocab"])

with open("all_res.json", "w") as outfile: 
    json.dump(all_res, outfile, indent=4)

# print(max([len(_) for _ in morphemes]))
