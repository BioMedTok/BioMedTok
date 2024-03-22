import os
import subprocess

import sentencepiece as spm

from transformers import CamembertTokenizer

f_in = open("./morpheme_fr_struct.txt","r")
morphemes = [e for e in f_in.read().split("\n") if len(e) > 3]
f_in.close()

infos = [
   ('./sources_preprocessed/CC100-FR_CHARS_lowercased_fixed_utf8.txt_morphemes-excluded', 'SentencePieceTokenizerMorphemesExcluded_CC100-FR_CHARS_lowercased_fixed_utf8_V4'),
   ('./sources_preprocessed/NACHOS_10M_lowercased_fixed_utf8.txt_morphemes-excluded', 'SentencePieceTokenizerMorphemesExcluded_NACHOS_10M_lowercased_fixed_utf8_V4'),
   ('./sources_preprocessed/PubMed_Abstracts_CHARS_lowercased_fixed_utf8.txt_morphemes-excluded', 'SentencePieceTokenizerMorphemesExcluded_PubMed_Abstracts_CHARS_lowercased_fixed_utf8_V4'),
   ('./sources_preprocessed/Wikipedia_CHARS_lowercased_fixed_utf8.txt_morphemes-excluded', 'SentencePieceTokenizerMorphemesExcluded_Wikipedia_CHARS_lowercased_fixed_utf8_V4'),
]

for info in infos:

   input=info[0]
   model_prefix=info[1]

   spm.SentencePieceTrainer.train(

      input=input,
      model_prefix=model_prefix,

      model_type='bpe',
      vocab_size=32000,
      train_extremely_large_corpus=True,
      num_threads=60,

      user_defined_symbols=morphemes,
   )

   output_tokenizer_path = f"./sentencepiece_tokenizers_morphemes_v4/{model_prefix}/"
   os.makedirs(output_tokenizer_path, exist_ok=True)
   print(output_tokenizer_path)

   vocab_file_path = f"./{model_prefix}.model"
   print(vocab_file_path)

   tokenizer = CamembertTokenizer(vocab_file=vocab_file_path, max_len=512)

   tokenizer.save_pretrained(output_tokenizer_path)

   subprocess.run(["mv", f"./{model_prefix}.vocab", output_tokenizer_path]) 


# import os
# import subprocess

# import sentencepiece as spm

# from transformers import CamembertTokenizer

# f_in = open("./morphemes_cleaned.csv","r")
# morphemes = [s.split(";")[0] for s in f_in.read().split("\n")]
# f_in.close()

# infos = [
#    ('./sources/CC100-FR_CHARS_lowercased_fixed_utf8.txt', 'SentencePieceTokenizer_CC100-FR_CHARS_lowercased_fixed_utf8_morphemes'),
#    ('./sources/NACHOS_10M_lowercased_fixed_utf8.txt', 'SentencePieceTokenizer_NACHOS_10M_lowercased_fixed_utf8_morphemes'),
#    ('./sources/PubMed_Abstracts_CHARS_lowercased_fixed_utf8.txt', 'SentencePieceTokenizer_PubMed_Abstracts_CHARS_lowercased_fixed_utf8_morphemes'),
#    ('./sources/Wikipedia_CHARS_lowercased_fixed_utf8.txt', 'SentencePieceTokenizer_Wikipedia_CHARS_lowercased_fixed_utf8_morphemes'),
# ]

# for info in infos:

#    input=info[0]
#    model_prefix=info[1]

#    spm.SentencePieceTrainer.train(

#       input=input,
#       model_prefix=model_prefix,

#       model_type='bpe',
#       vocab_size=32000,
#       train_extremely_large_corpus=True,
#       num_threads=60,

#       user_defined_symbols=morphemes,
#    )

#    output_tokenizer_path = f"./bpe_tokenizers/{model_prefix}/"
#    os.makedirs(output_tokenizer_path, exist_ok=True)
#    print(output_tokenizer_path)

#    vocab_file_path = f"./{model_prefix}.model"
#    print(vocab_file_path)

#    tokenizer = CamembertTokenizer(vocab_file=vocab_file_path, max_len=512)

#    tokenizer.save_pretrained(output_tokenizer_path)

#    subprocess.run(["mv", f"./{model_prefix}.vocab", output_tokenizer_path]) 
