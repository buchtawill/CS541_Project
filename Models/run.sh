#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=64g
#SBATCH -J "LITE_create_tensors"
#SBATCH -p short
#SBATCH -t 4:00:00
#SBATCH --mail-user=jwbuchta@wpi.edu
#SBATCH --mail-type=BEGIN,FAIL,END

module purge
module load slurm #cuda12.1 #python/3.12.4

now=$(date)
echo "INFO [run.sh] Starting execution on $now"

#source /home/jwbuchta/CS539_Project/Autoencoder/venv_autoencoder/bin/activate
#which $HOME/CS539_Project/Autoencoder/venv_autoencoder/bin/python
$HOME/CS539_Project/Autoencoder/venv_autoencoder/bin/python gen_spectrograms.py

#sleep 600

now=$(date)
echo "INFO [run.sh] Finished execution at $now"
