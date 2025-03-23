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
Notes: 
 - The comments and help messages in ```labrum_train.py``` and ```labrum_tune.py``` provide detailed explanations of each input argument
 - If you want to initialize a model with pre-trained weights, just pass the path to the model weights into the ```model_weights``` argument 
 - If you just want to do inference, pass the path to trained model weights into the ```model_weights``` argument and set ```num_epochs``` to 0
 - If you set the ```save_checkpoints``` argument in ```labrum_train.py``` to True, then weights will be saved each epoch and the code can resume from the latest checkpoint if training is interrupted (but beware of the storage cost; it is best to remove all the checkpoints after successfully training because they are no longer needed due to the highest val acc epoch weights being saved separately)
 - For cross-validation, the code will automatically handle resuming from the start of the latest fold/cycle if interrupted, but only if you pass a random seed into the ```seed``` argument (to ensure consistent splitting)
 - The default behavior is for the weights from the epoch with the highest validation accuracy to be retained for inference; these weights are saved regardless of if you have checkpointing on or not
 - Results will be automatically saved to CSV files and TensorBoard logs; exact predictions/logits and labels for each MRI are saved to ```{job_name}_probs.csv``` and ```{job_name}_val_probs.csv``` files to facilitate pushing to GitHub from an HPC, then pulling locally and creating figures in a Jupyter notebook

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
Notes: 
 - For hyperparameter tuning, model weights are only saved if they achieve above a certain threshold at inference (default is AUC > 0.70); for checkpointing, the Optuna study itself is saved and automatically resumed if you run the code again after being interrupted (but the model weights are not saved by default)

## Ensembling
 - Note that the performance metrics printed and logged to TensorBoard are all at the MRI sequence level. Each input MRI may contain several sequences, such as T1 coronal, T2 sagittal, etc.
 - We select which view (coronal, sagittal, or axial) to use using the ```view``` argument
 - For each view, we select which sequences to use using the ```sequence_type``` and ```fat_sat``` arguments (we use all available sequences for a given view in our paper, excluding localization/scout/survey sequences)
   - These arguments are ignored for the MRNet dataset as it only has one sequence per view
 - All dataset splits in this codebase are done at the _MRI_ID_ level, which ensures that sequences from the same original MRI are in the same split
 - The ```{job_name}_probs.csv``` and ```{job_name}_val_probs.csv``` results files can be used to aggregate predictions from the individual MRI sequences to the MRI_ID level (we simply average the sequence probabilities)
 - If you train separate models on sagittal, axial, and coronal sequences, you can then combine these final MRI_ID level probabilities (we simply average these across the three views)
 - [`ensemble.ipynb`](https://github.com/sahilsethi0105/scope-mri/blob/main/ensemble.ipynb) combines probabilities from models trained on separate views in the way specified above
   - If you prefer to use another method besides averaging to combine the sequence probabilities, change ```sequence_mode```
   - If you prefer to use another method besides averaging to combine the view probabilities, change ```view_mode```
  
## Summary of remaining files
 - ```MRI_and_metadata_import.py``` contains the logic
 - ```loader.py``` contains all data loading code
   - By default, we resize the MRI slices (to 256x256 for MRNet, and 400x400 for SCOPE-MRI), then center-crop to 224x224
   - We also normalize/standardize and apply augmentation in this file
   - We use the MONAI library for transformations (note that the library contains many additional medical imaging-specific transformations that can be added)
 - ```models.py``` contains the model architectures
   - For an individual MRI sequence, we pass each slice through the feature extractor 
 - ```semiexternal_eval.py``` is a file for running inference on an external test set. It follows the same logic as ```labrum_train.py``` when ```num_epochs=0```. 
