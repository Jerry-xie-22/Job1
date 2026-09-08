#!/bin/bash
#SBATCH --job-name=mintrec_generalize
#SBATCH --output=generalize_%j.log
#SBATCH --error=generalize_error_%j.log
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
cd /public/home/202420144954/job1/Base_generalize
python run.py \
    --dataset MIntRec \
    --method mag_bert \
    --data_mode multi-class \
    --train --save_results --tune \
    --seed 3 \
    --config_file_name mag_bert \
    --log_path logs/generalize \
    --results_path results_generalize \
    --results_file_name mintrec_to_mintrec20.csv \
    --video_feats_path video_feats.pkl \
    --audio_feats_path audio_feats.pkl \
    --test_video_feats_path video_feats2.pkl \
    --test_audio_feats_path audio_feats2.pkl \
    --text_backbone bert-large-uncased
