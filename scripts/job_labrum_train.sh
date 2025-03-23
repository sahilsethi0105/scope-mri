#!/bin/bash -l

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=40gb
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuq
#SBATCH --time=00-23:59:59

# Function to parse named arguments
parse_args() {
  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --preprocessed_folder) preprocessed_folder="$2"; shift ;;
      --label_column) label_column="$2"; shift ;;
      --batch_size) batch_size="$2"; shift ;;
      --num_epochs) num_epochs="$2"; shift ;;
      --model_type) model_type="$2"; shift ;;
      --job_name) job_name="$2"; shift ;;
      --lr) lr="$2"; shift ;;
      --weight_decay) weight_decay="$2"; shift ;;
      --dropout_rate) dropout_rate="$2"; shift ;;
      --augment) augment="$2"; shift ;;
      --augment_factor) augment_factor="$2"; shift ;;
      --augment_factor_0) augment_factor_0="$2"; shift ;;
      --model_weights) model_weights="$2"; shift ;;
      --transform_val) transform_val="$2"; shift ;;
      --ret_val_probs) ret_val_probs="$2"; shift ;;
      --view) view="$2"; shift ;;
      --scheduler) scheduler="$2"; shift ;;
      --sequence_type) sequence_type="$2"; shift ;;
      --fat_sat) fat_sat="$2"; shift ;;
      --contrast_or_no) contrast_or_no="$2"; shift ;;
      --dataset_type) dataset_type="$2"; shift ;;
      --pos_weight) pos_weight="$2"; shift ;;
      --script_mode) script_mode="$2"; shift ;;
      --n_cycles) n_cycles="$2"; shift ;;
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

source /gpfs/data/orthopedic-lab/ortho_ml/ortho_venv/bin/activate

# Run the Python script with parsed arguments
python3 labrum_train.py \
  --preprocessed_folder "$preprocessed_folder" \
  --label_column "$label_column" \
  --batch_size "$batch_size" \
  --num_epochs "$num_epochs" \
  --model_type "$model_type" \
  --job_name "$job_name" \
  --lr "$lr" \
  --weight_decay "$weight_decay" \
  --dropout_rate "$dropout_rate" \
  --augment "$augment" \
  --augment_factor "$augment_factor" \
  --augment_factor_0 "$augment_factor_0" \
  --model_weights "$model_weights" \
  --transform_val "$transform_val" \
  --ret_val_probs "${ret_val_probs:-True}" \
  --view "$view" \
  --scheduler "${scheduler:-ReduceLROnPlateau}" \
  --sequence_type "${sequence_type:-all}" \
  --fat_sat "${fat_sat:-all}" \
  --contrast_or_no "${contrast_or_no:-all}" \
  --dataset_type "$dataset_type" \
  --script_mode "${script_mode:-train}" \
  --n_cycles "${n_cycles:-5}" \
  --pos_weight "$pos_weight" > "${job_name}_results.txt"

deactivate

