#!/bin/bash

# Configuration
MODELS="gin gcn gat wlhn"
DATASETS="cora citeseer amazon-photo actor squirrel chameleon"
OUT_DIR="runs/tune_bgrl_wl_naive_cls_uniform_deeper_more_all_encoders_300ep_2026_04_15"

for m in $MODELS; do
    for d in $DATASETS; do
        echo "Lancement de $m sur $d..."
        
        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=tune_${m}_${d}
#SBATCH --output=logs/tune_${m}_${d}_%j.out
#SBATCH --error=logs/tune_${m}_${d}_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Charger ton environnement
module load anaconda3/2023.09-0/none-none
source /gpfs/softs/spack_1.0.2/opt/spack/linux-cascadelake/anaconda3-2023.09-0-v2nbar7o4nyuwoknqbnybvxufqw3rrnk/etc/profile.d/conda.sh
conda activate /gpfs/workdir/yartaouifa/.conda/envs/env_gssl

python3 -u -m wl_gcl.experiments.wl_dino.tune_wl_dino \
    --datasets $d \
    --model $m \
    --method bgrl_wl_naive_cls \
    --device cuda \
    --use_max_wl_depth \
    --search random \
    --n_trials 20 \
    --epochs 300 \
    --wl_naive_pair_sampling uniform \
    --wl_cls_levels all \
    --wl_cls_alpha_scheme deeper_more \
    --out_dir $OUT_DIR
EOT
    done
done