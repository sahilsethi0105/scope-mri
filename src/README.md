## Training
Filepaths to update in ```labrum_train.py```:
 - base_log_dir
 - csv_path
 - val_prob_csv_path
 - prob_csv_path

Example command line: 
```bash
python3 labrum_train.py \
  --preprocessed_folder '/path/to/directory/with/train_val_test_subdirectories/' \
  --label_column 'abnormal' \
  --batch_size 1 \
  --num_epochs 50 \
  --model_type AlexNet \
  --job_name alex_test1 \
  --lr 1e-3 \
  --weight_decay 1e-3 \
  --dropout_rate 0.1 \
  --augment True \
  --augment_factor 5 \
  --augment_factor_0 5 \
  --model_weights None \
  --transform_val True \
  --ret_val_probs True \
  --view axial \
  --scheduler 'ReduceLROnPlateau' \
  --sequence_type 'all' \
  --fat_sat 'all' \
  --contrast_or_no 'WO' \
  --dataset_type "MRNet" \
  --script_mode 'train' \
  --n_cycles 5 \
  --pos_weight 'automatic' 
```
Note that the comments and help messages in ```labrum_train.py``` and ```labrum_tune.py``` provide detailed explanations of each input argument.

## Tuning
Filepaths to update in ```labrum_tune.py```: 
 - log_dir
 - csv_path
 - prob_csv_path
 - study_save_path
 - study_trials_csv_path

Example command line (remember to adjust the trial params in ```labrum_tune.py``` as they can be set to override these input arguments): 
```bash
python3 labrum_tune.py \
  --preprocessed_folder '/path/to/directory/with/train_val_test_subdirectories/' \
  --label_column 'abnormal' \
  --model_weights None \
  --view 'axial' \
  --batch_size 1 \
  --num_epochs 50 \
  --model_type 'AlexNet \
  --job_name 'alex_tune1' \
  --augment True \
  --augment_factor 5 \
  --augment_factor_0 5 \
  --transform_val True \
  --ret_val_probs False \
  --n_trials 100 \
  --sequence_type 'all' \
  --fat_sat "all" \
  --contrast_or_no 'WO' \
  --dataset_type 'MRNet' \
  --pos_weight 'automatic' 
```


