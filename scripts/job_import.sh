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

source /gpfs/data/orthopedic-lab/ortho_ml/ortho_venv/bin/activate

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

# Count and print the number of files in the specified directory
file_count=$(ls /gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR\ 17404/preprocessed_MRIs/semiexternal_validation | wc -l)
echo "Number of files in the semiexternal_validation directory: $file_count"

deactivate




