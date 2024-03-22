
mkdir -p "./tokenized_dataset_morphemes_v4/SentencePieceTokenizerMorphemesExcluded_CC100-FR_CHARS_lowercased_fixed_utf8_V4/"
mkdir -p "./tokenized_dataset_morphemes_v4/SentencePieceTokenizerMorphemesExcluded_NACHOS_10M_lowercased_fixed_utf8_V4/"
mkdir -p "./tokenized_dataset_morphemes_v4/SentencePieceTokenizerMorphemesExcluded_PubMed_Abstracts_CHARS_lowercased_fixed_utf8_V4/"
mkdir -p "./tokenized_dataset_morphemes_v4/SentencePieceTokenizerMorphemesExcluded_Wikipedia_CHARS_lowercased_fixed_utf8_V4/"

echo "SentencePieceTokenizerMorphemesExcluded_CC100"

python preprocessing_dataset.py \
    --model_type='camembert' \
    --tokenizer_name="/users/ylabrak/TokenizationDrBERT/sentencepiece_tokenizers_morphemes_v4/SentencePieceTokenizerMorphemesExcluded_CC100-FR_CHARS_lowercased_fixed_utf8_V4/" \
    --train_file='./merged_7.45_Go_normalized_fixed_lowercased.txt' \
    --do_train \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --log_level='info' \
    --logging_first_step='True' \
    --cache_dir='./cache_dir/' \
    --path_save_dataset="./tokenized_dataset_morphemes_v4/SentencePieceTokenizerMorphemesExcluded_CC100-FR_CHARS_lowercased_fixed_utf8_V4/" \
    --output_dir='./test' \
    --preprocessing_num_workers=20

echo "SentencePieceTokenizerMorphemesExcluded_NACHOS_10M_lowercased_fixed_utf8_V4"

python preprocessing_dataset.py \
    --model_type='camembert' \
    --tokenizer_name="/users/ylabrak/TokenizationDrBERT/sentencepiece_tokenizers_morphemes_v4/SentencePieceTokenizerMorphemesExcluded_NACHOS_10M_lowercased_fixed_utf8_V4/" \
    --train_file='./merged_7.45_Go_normalized_fixed_lowercased.txt' \
    --do_train \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --log_level='info' \
    --logging_first_step='True' \
    --cache_dir='./cache_dir/' \
    --path_save_dataset="./tokenized_dataset_morphemes_v4/SentencePieceTokenizerMorphemesExcluded_NACHOS_10M_lowercased_fixed_utf8_V4/" \
    --output_dir='./test' \
    --preprocessing_num_workers=20

echo "SentencePieceTokenizerMorphemesExcluded_PubMed_Abstracts_CHARS_lowercased_fixed_utf8_V4"

python preprocessing_dataset.py \
    --model_type='camembert' \
    --tokenizer_name="/users/ylabrak/TokenizationDrBERT/sentencepiece_tokenizers_morphemes_v4/SentencePieceTokenizerMorphemesExcluded_PubMed_Abstracts_CHARS_lowercased_fixed_utf8_V4/" \
    --train_file='./merged_7.45_Go_normalized_fixed_lowercased.txt' \
    --do_train \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --log_level='info' \
    --logging_first_step='True' \
    --cache_dir='./cache_dir/' \
    --path_save_dataset="./tokenized_dataset_morphemes_v4/SentencePieceTokenizerMorphemesExcluded_PubMed_Abstracts_CHARS_lowercased_fixed_utf8_V4/" \
    --output_dir='./test' \
    --preprocessing_num_workers=20

echo "SentencePieceTokenizerMorphemesExcluded_Wikipedia_CHARS_lowercased_fixed_utf8_V4"

python preprocessing_dataset.py \
    --model_type='camembert' \
    --tokenizer_name="/users/ylabrak/TokenizationDrBERT/sentencepiece_tokenizers_morphemes_v4/SentencePieceTokenizerMorphemesExcluded_Wikipedia_CHARS_lowercased_fixed_utf8_V4/" \
    --train_file='./merged_7.45_Go_normalized_fixed_lowercased.txt' \
    --do_train \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --log_level='info' \
    --logging_first_step='True' \
    --cache_dir='./cache_dir/' \
    --path_save_dataset="./tokenized_dataset_morphemes_v4/SentencePieceTokenizerMorphemesExcluded_Wikipedia_CHARS_lowercased_fixed_utf8_V4/" \
    --output_dir='./test' \
    --preprocessing_num_workers=20
