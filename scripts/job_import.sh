#!/bin/bash -l

#SBATCH --cpus-per-task=4
#SBATCH --mem=512gb
#SBATCH --error=data_import.err
#SBATCH --output=data_import.out
#SBATCH --job-name=data_import
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=tier3q
#SBATCH --time=03:00:00

cd /gpfs/data/orthopedic-lab/ortho_ml 
pwd

module load gcc/12.1.0
module load python/3.10.5
module list 2>&1 

conda activate /gpfs/data/orthopedic-lab/ortho_env

pip list

which python

# Check CPU memory
echo "CPU Memory Info:"
free -h

echo ""

# Check GPU memory
if command -v nvidia-smi &> /dev/null
then
    echo "GPU Memory Info:"
    nvidia-smi
else
    echo "nvidia-smi could not be found. Are you sure you are on a GPU node?"
fi

echo ""

python3 MRI_and_metadata_import.py >data_import.txt

conda deactivate




