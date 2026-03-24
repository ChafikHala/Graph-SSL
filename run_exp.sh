#!/bin/bash
#SBATCH --job-name=wl_cora           # Nom de ton expérience
#SBATCH --output=logs/res_%j.out     # Fichier où s'écrira le texte (les print)
#SBATCH --error=logs/err_%j.err      # Fichier où s'écriront les erreurs
#SBATCH --partition=cpu_short        # La file d'attente (cpu_short pour des tests rapides)
#SBATCH --nodes=1                    # 1 nœud de calcul
#SBATCH --ntasks=1                   # 1 tâche
#SBATCH --cpus-per-task=4            # On demande 4 coeurs CPU pour que ça aille vite
#SBATCH --mem=8G                     # On demande 8 Go de RAM
#SBATCH --time=01:00:00              # Temps maximum alloué (Heures:Minutes:Secondes)

# 1. Charger les modules et l'environnement
module load anaconda3
source activate env_gssl

# 2. Se placer dans le bon dossier
cd $SLURM_SUBMIT_DIR

# 3. Lancer ton code Python
echo "Start of search..."

# Exhaustive grid search
#python wl_gcl/src/utils/tune.py --trainer wl_hierarchy --dataset cora --search grid

# Random search
python wl_gcl/src/utils/tune.py --trainer wl_hierarchy --dataset cora --search random --n_trials 20

echo "Search Ended"sbatch run_exp.sh