#!/bin/bash
#SBATCH --job-name=wl_cora           # Nom de ton expérience
#SBATCH --output=logs/res_%j.out     # Fichier où s'écrira le texte (les print)
#SBATCH --error=logs/err_%j.err      # Fichier où s'écriront les erreurs
#SBATCH --partition=cpu_short        # La file d'attente (cpu_short pour des tests rapides)
#SBATCH --nodes=1                    # 1 nœud de calcul
#SBATCH --ntasks=1                   # 1 tâche
#SBATCH --cpus-per-task=4            # On demande 4 coeurs CPU pour que ça aille vite
#SBATCH --mem=8G                     # On demande 8 Go de RAM
#SBATCH --time=02:00:00              # Temps maximum alloué (Heures:Minutes:Secondes)

# 1. Charger les modules et l'environnement
module load anaconda3
source activate env_gcl

# 2. Se placer dans le bon dossier
cd $SLURM_SUBMIT_DIR

# 3. Lancer ton code Python
# Ici on appelle la méthode wl_hierarchy sur le dataset cora avec le modèle gin
echo "Lancement de l'expérience..."
python main.py --method wl_hierarchy --dataset cora --model gin