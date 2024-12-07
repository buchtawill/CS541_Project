#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64g
#SBATCH -J "fma_higher_res"
#SBATCH -p short
#SBATCH -t 2:00:00
#SBATCH --mail-user=jwbuchta@wpi.edu
#SBATCH --mail-type=BEGIN,FAIL,END

module purge
module load slurm #cuda12.1 #python/3.12.4

now=$(date)
echo "INFO [run.sh] Starting execution on $now"

#source /home/jwbuchta/CS539_Project/Autoencoder/venv_autoencoder/bin/activate
#which $HOME/CS539_Project/Autoencoder/venv_autoencoder/bin/python
# $HOME/CS539_Project/Autoencoder/venv_autoencoder/bin/python get_song_y.py
$HOME/CS539_Project/Autoencoder/venv_autoencoder/bin/python gen_spec_tensors.py
# tar czf fma_medium.tar.gz ./fma_medium

#sleep 600

now=$(date)
echo "INFO [run.sh] Finished execution at $now"
