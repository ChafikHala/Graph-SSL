#!/bin/bash
#SBATCH --job-name=wl_amazon        # Nom de ton expérience
#SBATCH --output=logs/res_%j.out     # Fichier où s'écrira le texte (les print)
#SBATCH --error=logs/err_%j.err      # Fichier où s'écriront les erreurs
#SBATCH --partition=gpu        # La file d'attente 
#SBATCH --gres=gpu:1                   # 1 nœud de calcul
#SBATCH --cpus-per-task=4            # On demande 4 coeurs CPU pour que ça aille vite
#SBATCH --mem=24G                     # Go de RAM
#SBATCH --time=01:00:00              # Temps maximum alloué (Heures:Minutes:Secondes)

# 1. Charger les modules et l'environnemento 
module load anaconda3/2023.09-0/none-none
source activate env_gssl

# 2. Se placer dans le bon dossier
cd $SLURM_SUBMIT_DIR

# 3. Lancer ton code Python
echo "Start of search..."


# Add path
export PYTHONPATH=${PYTHONPATH}:${SLURM_SUBMIT_DIR}


# Exhaustive grid search
#python wl_gcl/src/utils/tune.py --trainer wl_hierarchy --dataset cora --search grid

# Random search
python -u wl_gcl/src/utils/tune.py --trainer wl_hierarchy --dataset amazon-photo --search random --n_trials 40 --device cuda

echo "Search Ended"

