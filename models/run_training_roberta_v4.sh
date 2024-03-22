#!/bin/bash

#SBATCH --job-name=BertTok     # nom du job
#SBATCH --constraint=v100-32g
#SBATCH --ntasks=32                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=4          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:4                 # nombre de GPU par nœud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (un quart du noeud ici)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=./logs_slurm/mlm_test%j.out # nom du fichier de sortie
#SBATCH --error=./logs_slurm/mlm_test%j.err  # nom du fichier d'erreur (ici commun avec la sortie)
#
# Envoi des mails
#SBATCH --mail-type=begin,fail,abort,end
 
# Nettoyage des modules charges en interactif et herites par defaut
module purge
 
# Chargement des modules
module load pytorch-gpu/py3/1.12.1
# module load pytorch-gpu/py3/1.7.1+nccl-2.8.3-1
 
# Echo des commandes lancees
set -x -e

export OMP_NUM_THREADS=10

export CUDA_LAUNCH_BLOCKING=1

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

# rm -rf /gpfsscratch/rech/rtl/uaj63yz/FilesTokenizers/
# cp -r FilesTokenizers /gpfsscratch/rech/rtl/uaj63yz/

# BPE HF No Morphemes
# sbatch -A rtl@v100 run_training_roberta_v1.sh './tokenizers/BpeTokenizer_Wikipedia_CHARS_lowercased_fixed_utf8-roberta/' "./tokenized_datasets/BpeTokenizer_Wikipedia_CHARS_lowercased_fixed_utf8-roberta/" './models_v2/ModelBpeTokenizer_Wikipedia_CHARS_lowercased_fixed_utf8-roberta/' './logs_v2/ModelBpeTokenizer_Wikipedia_CHARS_lowercased_fixed_utf8-roberta/'
# sbatch -A rtl@v100 run_training_roberta_v1.sh './tokenizers/BpeTokenizer_CC100-FR_CHARS_lowercased_fixed_utf8-roberta/' "./tokenized_datasets/BpeTokenizer_CC100-FR_CHARS_lowercased_fixed_utf8-roberta/" './models_v2/ModelBpeTokenizer_CC100-FR_CHARS_lowercased_fixed_utf8-roberta/' './logs_v2/ModelBpeTokenizer_CC100-FR_CHARS_lowercased_fixed_utf8-roberta/'
# sbatch -A rtl@v100 run_training_roberta_v1.sh './tokenizers/BpeTokenizer_NACHOS_10M_lowercased_fixed_utf8-roberta/' "./tokenized_datasets/BpeTokenizer_NACHOS_10M_lowercased_fixed_utf8-roberta/" './models_v2/ModelBpeTokenizer_NACHOS_10M_lowercased_fixed_utf8-roberta/' './logs_v2/ModelBpeTokenizer_NACHOS_10M_lowercased_fixed_utf8-roberta/'
# sbatch -A rtl@v100 run_training_roberta_v1.sh './tokenizers/BpeTokenizer_PubMed_Abstracts_CHARS_lowercased_fixed_utf8-roberta/' "./tokenized_datasets/BpeTokenizer_PubMed_Abstracts_CHARS_lowercased_fixed_utf8-roberta/" './models_v2/ModelBpeTokenizer_PubMed_Abstracts_CHARS_lowercased_fixed_utf8-roberta/' './logs_v2/ModelBpeTokenizer_PubMed_Abstracts_CHARS_lowercased_fixed_utf8-roberta/'

tokenizer_in=$1
dataset_tokenized_in=$2
model_out=$3
logs_out=$4

mkdir -p "${model_out}"
chmod -R 777 "${model_out}"

mkdir -p "${logs_out}"
chmod -R 777 "${logs_out}"

echo "${tokenizer_in}"
echo "${dataset_tokenized_in}"
echo "${model_out}"
echo "${logs_out}"

srun -l python -u run_training_roberta_v1.py \
    --model_type='camembert' \
    --config_overrides="max_position_embeddings=514,type_vocab_size=1,vocab_size=32000,bos_token_id=0,pad_token_id=1,eos_token_id=2" \
    --tokenizer_name="${tokenizer_in}" \
    --path_load_dataset="${dataset_tokenized_in}" \
    --output_dir="${model_out}" \
    --logging_dir="${logs_out}" \
    --per_device_train_batch_size=32 \
    --do_train \
    --warmup_steps=10000 \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --logging_steps=500 \
    --report_to='tensorboard' \
    --save_strategy='epoch' \
    --skip_memory_metrics='False' \
    --log_level='info' \
    --logging_first_step='True' \
    --num_train_epochs=400 \
    --fp16 \
    --save_total_limit=400 \
    --ddp_timeout=600 \
    --ddp_find_unused_parameters='False' \
