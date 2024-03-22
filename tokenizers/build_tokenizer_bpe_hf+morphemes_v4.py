import os

from tokenizers import ByteLevelBPETokenizer

from transformers import CamembertTokenizer, RobertaTokenizer

f_in = open("./morpheme_fr_struct.txt","r")
morphemes = [e for e in f_in.read().split("\n") if len(e) > 3]
f_in.close()

infos = [
   ('./sources_preprocessed/CC100-FR_CHARS_lowercased_fixed_utf8.txt_morphemes-excluded', 'BpeTokenizerMorphemesExcluded_CC100-FR_CHARS_lowercased_fixed_utf8_v5'),
   ('./sources_preprocessed/NACHOS_10M_lowercased_fixed_utf8.txt_morphemes-excluded', 'BpeTokenizerMorphemesExcluded_NACHOS_10M_lowercased_fixed_utf8_v5'),
   ('./sources_preprocessed/PubMed_Abstracts_CHARS_lowercased_fixed_utf8.txt_morphemes-excluded', 'BpeTokenizerMorphemesExcluded_PubMed_Abstracts_CHARS_lowercased_fixed_utf8_v5'),
   ('./sources_preprocessed/Wikipedia_CHARS_lowercased_fixed_utf8.txt_morphemes-excluded', 'BpeTokenizerMorphemesExcluded_Wikipedia_CHARS_lowercased_fixed_utf8_v5'),
]

for info in infos:

   input=info[0]
   model_prefix=info[1]

   tokenizer = ByteLevelBPETokenizer()
   
   # tokenizer.add_special_tokens(["<mask>"])

   # tokenizer.train(files=input, vocab_size=32000, min_frequency=2)
   # tokenizer.add_special_tokens(["<s>","<pad>", "</s>", "<unk>", "<mask>"] + [f"<{m}>" for m in morphemes[0:800]])

   # tokenizer.train(files=input, vocab_size=32000, min_frequency=2, special_tokens=[
   #    "<s>",
   #    "<pad>",
   #    "</s>",
   #    "<unk>",
   #    "<mask>",
   # ])
   tokenizer.train(files=input, vocab_size=32000, min_frequency=2, special_tokens=[
      "<s>",
      "<pad>",
      "</s>",
      "<unk>",
      "<mask>",
   ] + [f"{m}" for m in morphemes[0:800]])
   # ] + [f"<{m}>" for m in morphemes[0:800]])

   # 100 is OK
   # 500 is OK
   # 800 is OK
   print(morphemes[0:800])
   # ] + morphemes)

   output_path = f"./bpe_tokenizers_morphemes_v5/{model_prefix}"
   os.makedirs(output_path, exist_ok=True)
   # tokenizer.add_special_tokens(["<s>","<pad>", "</s>", "<unk>", "<mask>"] + [f"{m}" for m in morphemes[0:800]])
   tokenizer.save_model(output_path, model_prefix)
   
   tokenizer = RobertaTokenizer(f"{output_path}/{model_prefix}-vocab.json", f"{output_path}/{model_prefix}-merges.txt")
   # tokenizer.add_special_tokens({f"{m}_token": f"{m}" for m in morphemes[0:800]})

   special_tokens_dict = {'additional_special_tokens': [f"{m}" for m in morphemes[0:800]]}
   num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
   print(f"Add {num_added_toks} tokens")
   
   tokenizer_roberta_path = f"./bpe_tokenizers_morphemes_v5/{model_prefix}-roberta/"
   os.makedirs(tokenizer_roberta_path, exist_ok=True)
   tokenizer.save_pretrained(tokenizer_roberta_path)

   # print("-"*50)

   # tokenizer_camembert = CamembertTokenizer.from_pretrained(tokenizer_roberta_path, max_len=512)
   # print(tokenizer_camembert)
   # tokenizer_camembert = CamembertTokenizer(vocab_file=tokenizer_roberta_path, max_len=512)

   # tokenizer_camembert_path = f"./bpe_tokenizers/{model_prefix}-camembert/"
   # os.makedirs(tokenizer_camembert_path, exist_ok=True)

   # tokenizer_camembert.save_pretrained(tokenizer_camembert_path)


# import os

# from tokenizers import ByteLevelBPETokenizer

# from transformers import CamembertTokenizer, RobertaTokenizer

# f_in = open("./morphemes_cleaned.csv","r")
# morphemes = [s.split(";")[0] for s in f_in.read().split("\n")]
# f_in.close()

# infos = [
#    ('./sources/CC100-FR_CHARS_lowercased_fixed_utf8.txt', 'BpeTokenizer_CC100-FR_CHARS_lowercased_fixed_utf8_morphemes'),
#    ('./sources/NACHOS_10M_lowercased_fixed_utf8.txt', 'BpeTokenizer_NACHOS_10M_lowercased_fixed_utf8_morphemes'),
#    ('./sources/PubMed_Abstracts_CHARS_lowercased_fixed_utf8.txt', 'BpeTokenizer_PubMed_Abstracts_CHARS_lowercased_fixed_utf8_morphemes'),
#    ('./sources/Wikipedia_CHARS_lowercased_fixed_utf8.txt', 'BpeTokenizer_Wikipedia_CHARS_lowercased_fixed_utf8_morphemes'),
# ]

# for info in infos:

#    input=info[0]
#    model_prefix=info[1]

#    tokenizer = ByteLevelBPETokenizer()

#    tokenizer.train(files=input, vocab_size=32000, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"] + morphemes)

#    output_path = f"./bpe_tokenizers/{model_prefix}"
#    os.makedirs(output_path, exist_ok=True)
#    tokenizer.save_model(output_path, model_prefix)

#    tokenizer = RobertaTokenizer(f"{output_path}/{model_prefix}-vocab.json", f"{output_path}/{model_prefix}-merges.txt")
   
#    tokenizer_roberta_path = f"./bpe_tokenizers/{model_prefix}-roberta-morphemes/"
#    os.makedirs(tokenizer_roberta_path, exist_ok=True)
#    tokenizer.save_pretrained(tokenizer_roberta_path)

#    # print("-"*50)

#    # tokenizer_camembert = CamembertTokenizer.from_pretrained(tokenizer_roberta_path, max_len=512)
#    # print(tokenizer_camembert)
#    # tokenizer_camembert = CamembertTokenizer(vocab_file=tokenizer_roberta_path, max_len=512)

#    # tokenizer_camembert_path = f"./bpe_tokenizers/{model_prefix}-camembert/"
#    # os.makedirs(tokenizer_camembert_path, exist_ok=True)

#    # tokenizer_camembert.save_pretrained(tokenizer_camembert_path)
