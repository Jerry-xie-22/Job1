#!/bin/bash
#SBATCH --job-name=job2
#SBATCH --output=seed89_%j.log
#SBATCH --error=seed89_error%j.log
#SBATCH --partition=gpuA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu4
NODE=$(hostname)
echo "===== 当前任务运行节点：$NODE ====="

# 新增：限制MKL高级指令集，解决Illegal instruction
export MKL_DEBUG_CPU_TYPE=5
export MKL_ENABLE_INSTRUCTIONS=SSE4_2

# 新增：限制数学库线程数，和cpus-per-task匹配，防止多线程内存冲突
export OMP_NUM_THREADS=9
export MKL_NUM_THREADS=9


# # 让PyTorch不要使用不安全的CPU指令
# export LRU_CACHE_CAPACITY=1

# 新增：放开core文件大小，方便后续调试段错误
ulimit -c unlimited
echo "=== 当前运行节点 ==="
hostname

echo "=== CPU 型号信息 ==="
lscpu | grep "Model name"

echo "=== CPU 支持的指令集（检查是否有 avx2, avx512） ==="
cat /proc/cpuinfo | grep flags | head -n 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mvcl-daf

for dataset in "MIntRec"
do
    for seed in 3
    do
        python /public/home/202420144954/job1/Base/run.py\
            --dataset $dataset \
            --method 'mag_bert' \
            --data_mode 'multi-class' \
            --train \
            --save_results \
            --tune \
            --seed $seed \
            --logger_name 'intra_inter_fusion' \
            --log_path '/public/home/202420144954/job1/Base/logs/rerun' \
            --config_file_name 'mag_bert' \
            --results_file_name 'results_ablation.csv' \
            --results_path '/public/home/202420144954/job1/Base/results_ablation' \
            --video_feats_path 'video_feats.pkl' \
            --audio_feats_path 'audio_feats.pkl' \
            --text_backbone 'bert-large-uncased'
    done
done
