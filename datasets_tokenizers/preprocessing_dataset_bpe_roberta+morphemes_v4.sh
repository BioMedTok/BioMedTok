
mkdir -p "./tokenized_dataset_morphemes_bpe_v5/BpeTokenizerMorphemesExcluded_CC100-FR_CHARS_lowercased_fixed_utf8_v5-roberta/"
mkdir -p "./tokenized_dataset_morphemes_bpe_v5/BpeTokenizerMorphemesExcluded_NACHOS_10M_lowercased_fixed_utf8_v5-roberta/"
mkdir -p "./tokenized_dataset_morphemes_bpe_v5/BpeTokenizerMorphemesExcluded_PubMed_Abstracts_CHARS_lowercased_fixed_utf8_v5-roberta/"
mkdir -p "./tokenized_dataset_morphemes_bpe_v5/BpeTokenizerMorphemesExcluded_Wikipedia_CHARS_lowercased_fixed_utf8_v5-roberta/"

echo "SentencePieceTokenizerMorphemesExcluded_CC100"

python preprocessing_dataset_roberta.py \
    --model_type='camembert' \
    --tokenizer_name="/users/ylabrak/TokenizationDrBERT/bpe_tokenizers_morphemes_v5/BpeTokenizerMorphemesExcluded_CC100-FR_CHARS_lowercased_fixed_utf8_v5-roberta/" \
    --train_file='./merged_7.45_Go_normalized_fixed_lowercased.txt' \
    --do_train \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --log_level='info' \
    --logging_first_step='True' \
    --cache_dir='./cache_dir/' \
    --path_save_dataset="./tokenized_dataset_morphemes_bpe_v5/BpeTokenizerMorphemesExcluded_CC100-FR_CHARS_lowercased_fixed_utf8_v5-roberta/" \
    --output_dir='./test' \
    --preprocessing_num_workers=20

echo "BpeTokenizerMorphemesExcluded_NACHOS_10M_lowercased_fixed_utf8"

python preprocessing_dataset_roberta.py \
    --model_type='camembert' \
    --tokenizer_name="/users/ylabrak/TokenizationDrBERT/bpe_tokenizers_morphemes_v5/BpeTokenizerMorphemesExcluded_NACHOS_10M_lowercased_fixed_utf8_v5-roberta/" \
    --train_file='./merged_7.45_Go_normalized_fixed_lowercased.txt' \
    --do_train \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --log_level='info' \
    --logging_first_step='True' \
    --cache_dir='./cache_dir/' \
    --path_save_dataset="./tokenized_dataset_morphemes_bpe_v5/BpeTokenizerMorphemesExcluded_NACHOS_10M_lowercased_fixed_utf8_v5-roberta/" \
    --output_dir='./test' \
    --preprocessing_num_workers=20

echo "BpeTokenizerMorphemesExcluded_PubMed_Abstracts_CHARS_lowercased_fixed_utf8"

python preprocessing_dataset_roberta.py \
    --model_type='camembert' \
    --tokenizer_name="/users/ylabrak/TokenizationDrBERT/bpe_tokenizers_morphemes_v5/BpeTokenizerMorphemesExcluded_PubMed_Abstracts_CHARS_lowercased_fixed_utf8_v5-roberta/" \
    --train_file='./merged_7.45_Go_normalized_fixed_lowercased.txt' \
    --do_train \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --log_level='info' \
    --logging_first_step='True' \
    --cache_dir='./cache_dir/' \
    --path_save_dataset="./tokenized_dataset_morphemes_bpe_v5/BpeTokenizerMorphemesExcluded_PubMed_Abstracts_CHARS_lowercased_fixed_utf8_v5-roberta/" \
    --output_dir='./test' \
    --preprocessing_num_workers=20

echo "BpeTokenizerMorphemesExcluded_Wikipedia_CHARS_lowercased_fixed_utf8"

python preprocessing_dataset_roberta.py \
    --model_type='camembert' \
    --tokenizer_name="/users/ylabrak/TokenizationDrBERT/bpe_tokenizers_morphemes_v5/BpeTokenizerMorphemesExcluded_Wikipedia_CHARS_lowercased_fixed_utf8_v5-roberta/" \
    --train_file='./merged_7.45_Go_normalized_fixed_lowercased.txt' \
    --do_train \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --log_level='info' \
    --logging_first_step='True' \
    --cache_dir='./cache_dir/' \
    --path_save_dataset="./tokenized_dataset_morphemes_bpe_v5/BpeTokenizerMorphemesExcluded_Wikipedia_CHARS_lowercased_fixed_utf8_v5-roberta/" \
    --output_dir='./test' \
    --preprocessing_num_workers=20
