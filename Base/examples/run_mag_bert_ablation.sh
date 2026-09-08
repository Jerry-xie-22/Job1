#!/bin/bash
#SBATCH --job-name=base_ablation
#SBATCH --output=base_ablation_%j.log
#SBATCH --error=base_ablation_error_%j.log
#SBATCH --partition=gpuA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu4

set -e
export MKL_DEBUG_CPU_TYPE=5
export MKL_ENABLE_INSTRUCTIONS=SSE4_2
export OMP_NUM_THREADS=9
export MKL_NUM_THREADS=9

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mvcl-daf
cd /public/home/202420144954/job1/Base

python run.py \
    --dataset MIntRec \
    --method mag_bert \
    --data_mode multi-class \
    --train --save_results --tune --seed 3 \
    --config_file_name mag_bert_ablation \
    --log_path logs/ablation_components \
    --results_path results_ablation/components \
    --results_file_name component_ablation.csv \
    --video_feats_path video_feats.pkl \
    --audio_feats_path audio_feats.pkl \
    --label_descriptions_path /public/home/202420144954/job2/Base_for_emo_mintrec10_c2f_inject_large/data/label_descriptions_mintrec.pt \
    --text_backbone bert-large-uncased
