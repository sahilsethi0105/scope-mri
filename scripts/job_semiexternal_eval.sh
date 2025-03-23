#!/bin/bash -l

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=40gb
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuq
#SBATCH --time=00-18:00:00

# Function to parse named arguments
parse_args() {
  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --base_folder) base_folder="$2"; shift ;;
      --model_weights) model_weights="$2"; shift ;;
      --label_column) label_column="$2"; shift ;;
      --model_type) model_type="$2"; shift ;;
      --batch_size) batch_size="$2"; shift ;;
      --contrast_or_no) contrast_or_no="$2"; shift ;;
      --view) view="$2"; shift ;;
      --num_workers) num_workers="$2"; shift ;;
      --job_name) job_name="$2"; shift ;;
      *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
  done
}

export TORCH_HOME=/gpfs/data/orthopedic-lab/torch_cache

# Parse the arguments
parse_args "$@"

cd /gpfs/data/orthopedic-lab/ortho_ml

module purge
module load gcc/12.1.0
module load python/3.10.5
module list 2>&1

conda activate /gpfs/data/orthopedic-lab/ortho_env 

# Run the Python script with parsed arguments
python3 semiexternal_eval.py \
  --base_folder "$base_folder" \
  --model_weights "$model_weights" \
  --label_column "$label_column" \
  --model_type "$model_type" \
  --view "$view" \
  --batch_size "${batch_size:-1}" \
  --contrast_or_no "${contrast_or_no:-all}" \
  --num_workers "${num_workers:-4}" \
  --job_name "$job_name" > "${job_name}_results.txt"

conda deactivate

