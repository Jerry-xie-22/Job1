#!/bin/bash
#SBATCH --job-name=mintrec20_generalize
#SBATCH --output=generalize20_%j.log
#SBATCH --error=generalize20_error_%j.log
#SBATCH --partition=gpuA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --exclude=gpu4
set -e
export MKL_DEBUG_CPU_TYPE=5
export MKL_ENABLE_INSTRUCTIONS=SSE4_2
export OMP_NUM_THREADS=9
export MKL_NUM_THREADS=9
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mvcl-daf
cd /public/home/202420144954/job1/Basefor2.0_generalize
# Preserves Basefor2.0's grid: 162 combinations per seed.
# Set the config lists to your source-dev-selected values for a single experiment.
python run.py \
    --dataset MIntRec2.0 \
    --method mag_bert --data_mode multi-class \
    --train --save_results --tune --seed 2 \
    --config_file_name mag_bert \
    --log_path logs/generalize \
    --results_path results_generalize \
    --results_file_name mintrec20_to_mintrec.csv \
    --video_feats_path video_feats.pkl \
    --audio_feats_path audio_feats.pkl \
    --test_video_feats_path video_feats1.pkl \
    --test_audio_feats_path audio_feats1.pkl \
    --text_backbone bert-large-uncased
