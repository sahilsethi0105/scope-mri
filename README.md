# SCOPE-MRI: Intro to deep learning for MRIs/CTs
This repository is meant to be a useful resource for getting started with using deep learning on MRIs and CT scans. It accompanies the release of the Shoulder Comprehensive Orthopedic Pathology Evaluation (SCOPE)-MRI dataset—a publicly available shoulder MRI dataset with image-level labels for several different pathologies. The dataset is described in our paper:

> [**SCOPE-MRI: Bankart Lesion Detection as a Case Study in Data Curation and Deep Learning for Challenging Diagnoses**](https://www.nature.com/articles/s44387-025-00043-5)<br/>
  Sahil Sethi, Sai Reddy, Mansi Sakarvadia, Jordan Serotte, Darlington Nwaudo, Nicholas Maassen, & Lewis Shi. <b>Nature Partner Journal (NPJ) Artificial Intelligence</b>, 2025.

This is an extension of the work described in our prior paper:

> [**Toward Non-Invasive Diagnosis of Bankart Lesions with Deep Learning**](https://doi.org/10.1117/12.3046251)<br/>
  Sahil Sethi, Sai Reddy, Mansi Sakarvadia, Jordan Serotte, Darlington Nwaudo, Nicholas Maassen, & Lewis Shi. <b>Proc. SPIE 13407, Medical Imaging 2025: Computer-Aided Diagnosis, 134073G</b>.

Although the repo was developed for the above papers, we have written the code so that it can be easily adapted for training, hyperparameter tuning, cross-validating, and using GradCAM for any binary classification task using MRIs or CT scans.
 - The information in [`src/README.md`](https://github.com/sahilsethi0105/scope-mri/tree/main/src#readme), in [`grad_cam/README.md`](https://github.com/sahilsethi0105/scope-mri/blob/main/grad_cam/README.md), and below covers getting started with the repo for either the Stanford MRNet knee MRI dataset or our SCOPE-MRI dataset
 - If you have your own dataset, [`MRI_and_metadata_import.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/MRI_and_metadata_import.py) and [`train_test_val_creation.py`](https://github.com/sahilsethi0105/ortho_ml/blob/main/train_test_val_creation.py) show how to go from the raw DICOM files to a final, preprocessed dataset; follow the instructions below for using the SCOPE-MRI dataset, but make the following adjustments:
   - In the ```main()``` function and ```parser.add_argument()``` statements of [`train_test_val_creation.py`](https://github.com/sahilsethi0105/ortho_ml/blob/main/train_test_val_creation.py), adjust the number of labels based on how many your dataset has (eg, if only one binary label, then remove ```args.label_column2``` and ```args.label_column3```); the split will be stratified by these labels
   - Specify a random seed for ```random_state``` in ```stratified_split_data_by_mri_id()``` if desired (eg, 42); also adjust ```val_size``` and ```test_size``` as desired (default uses 70% train / 10% val / 20% test)
   - In [`loader.py`](https://github.com/sahilsethi0105/ortho_ml/blob/main/loader.py), adjust transformations/augmentations and preprocessing as desired (we currently center-crop each slice to focus on the region of interest)
 - If you prefer to customize your dataset structure and dataloader code, do so in [`loader.py`](https://github.com/sahilsethi0105/ortho_ml/blob/main/loader.py)
     - Include any desired additional preprocessing and/or augmentation code)
     - Update ```prepare_and_create_loaders()``` and ```prepare_and_create_loaders_from_params()``` accordingly (they are used for [`labrum_train.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_train.py)  and [`labrum_tune.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_tune.py), respectively)
     - Many input arguments for [`labrum_train.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_train.py) and [`labrum_tune.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_tune.py) are specific for our SCOPE-MRI dataset (called ```labrum``` in the ```dataset_type``` argument); you can remove and/or modify these based on your code for preparing dataloaders
 - You can add custom models or modify the existing ones in [`models.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/models.py)
 - This version of the codebase is designed to train all models on a single GPU, but manually implements much of the same behavior as PyTorch Lightning (eg, checkpointing, Optuna integration, TensorBoard logging); if you wish to train on multiple GPUs, we recommend using [`this article`](https://lightning.ai/docs/pytorch/stable/starter/converting.html) to adapt the code 
   - You will also need to change [`job_labrum_train.sh`](https://github.com/sahilsethi0105/scope-mri/blob/main/scripts/job_labrum_train.sh) to request multiple GPUs on your HPC
 - For multi-class classification, you will need to update the loss function and classifier parts of the models accordingly, as well as possibly modify some of the logging

## Installation

Requirements:

- `python==3.10`

```bash
git clone https://github.com/sahilsethi0105/scope-mri.git
cd scope-mri
conda env create -f environment.yml
conda activate ortho_env 
```

## Accessing the Data
If you simply want to use this repo to get familiar with deep learning for MRIs/CTs, we recommend installing the [`Stanford MRNet dataset available here`](https://stanfordmlgroup.github.io/competitions/mrnet/). Their dataset is larger than ours and significantly easier to train on. The repo is already compatible with MRNet. 

However, if you are interested in the SCOPE-MRI dataset, it has been released on the [`Medical Imaging and Data Resource Center (MIDRC)`](https://www.midrc.org/). Non-commercial access is freely available per MIDRC's usage policies to government and academic researchers. You can search for our MRIs in their system using the `UChicago-ScopeMRI` tag, or access the dataset directly via [`this link`](https://www.nature.com/articles/s44387-025-00043-5). _**WILL BE UPDATED WITH FURTHER INSTRUCTIONS SHORTLY**_. Then, follow the data preprocessing steps below. 

## Using the Repo with SCOPE-MRI
- [`MRI_and_metadata_import.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/MRI_and_metadata_import.py):
- [`train_test_val_creation.py`](https://github.com/sahilsethi0105/ortho_ml/blob/main/train_test_val_creation.py):
- For all files in this codebase, **your ```preprocessed_folder``` should be the final folder that contains your ```train```, ```val```, and ```test``` subfolder**
  - Note that these each contain subfolders for each MRI_ID, each with one preprocessed .npy array for each sequence in that MRI
- Note that we manually filtered out sequences that were repeated (the repeat was kept and original was removed, with the assumption that the poor image quality let to the repreat sequence) _**WILL BE UPDATED WITH EXACT Instructions SHORTLY**_
- _**WILL BE UPDATED WITH How to Get Labels and Metadata SHORTLY**_
- Specify a random seed for ```random_state``` in ```stratified_split_data_by_mri_id()``` if desired (eg, 42)

## Using the Repo with MRNet
 - First, fill out the dataset research use agreement with your email [`here`](https://stanfordmlgroup.github.io/competitions/mrnet/), and you should automatically receive a link to download the data 
 - If they are no longer maintaining that website, Stanford has also posted it [`here`](https://aimi.stanford.edu/datasets/mrnet-knee-mris)
 - After unzipping the folder, you should see ```train``` and ```valid``` subfolders
     - Our code uses the `valid` set as a hold-out test set, and dynamically selects a 120-MRI subset of the ```train``` data to monitor progresss as a validation/tuning set
     - You can adjust this by changing ```create_stratified_validation_set()``` and when it is called in ```prepare_datasets()``` in [`loader.py`](https://github.com/sahilsethi0105/ortho_ml/blob/main/loader.py)
 - Their dataset contains three binary labels: 'acl', 'meniscus', and 'abnormal'
     - Labels for each are found in the corresponding CSVs for each spit (eg, train-abnormal.csv and val-abnormal.csv for the 'abnormal' label, which is what we use for pre-training)
 - Simply **pass in the path to the base folder that contains the original ```train``` and ```valid``` subfolders for the ```preprocessed_folder``` argument** in all of the files in this codebase, and the files should all run properly
   - Make sure to adjust the other input arguments as desired (eg, ``model_type``, ``view``, etc.)
   - Arguments specific to SCOPE-MRI, such as ```sequence_type``` and ```contrast_or_no``` will be ignored, so you can set them to any valid value

## Visualizing MRIs
 - [`visualize_MRIs.ipynb`](https://github.com/sahilsethi0105/scope-mri/blob/main/visualize_MRIs.ipynb) is a Jupyter notebook for viewing the MRIs
 - For SCOPE-MRI: it provides code for viewing a target slice from all available sequences for a specific target MRI_ID
     - For ``base_path``, pass in the same directory used for the ```preprocessed_folder``` argument in the other files
 - For MRNet: it provides code for viewing a target slice from the coronal, sagittal, and axial views for a specific target MRI_ID
 - For whichever dataset you are using, pass in the corresponding ```preprocessed_folder``` as the ```base_path``` argument here

## Training, Cross-Validation, Hyperparameter Tuning, and Ensembling
- [`labrum_train.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_train.py): trains models, does cross-validation, and does inference using either MRNet data or SCOPE-MRI
- [`labrum_tune.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_tune.py): tunes models using either MRNet data or SCOPE-MRI
- [`ensemble.ipynb`](https://github.com/sahilsethi0105/scope-mri/blob/main/ensemble.ipynb): combines models trained on separate views, and provides performance metrics at the MRI_ID level
- See [`src/README.md`](https://github.com/sahilsethi0105/scope-mri/tree/main/src#readme) for additional information

## Grad-CAM: Interpreting what a model learned
- [`grad_cam_med.py`](https://github.com/sahilsethi0105/scope-mri/blob/grad_cam/grad_cam/grad_cam_med.py): outputs Grad-CAM heat maps of what the model is "looking at" in each image (one heatmap for each slice in each MRI in the test set)
- See [`grad_cam/README.md`](https://github.com/sahilsethi0105/scope-mri/blob/main/grad_cam/README.md) for additional information

## Additional Notes
 - The commands in [`src/README.md`](https://github.com/sahilsethi0105/scope-mri/tree/main/src#readme) are for directly running the files
 - [`scripts/`](https://github.com/sahilsethi0105/scope-mri/tree/main/scripts) contains the shell scripts used to submit jobs to SLURM if using an HPC
 - The files in [`src/`](https://github.com/sahilsethi0105/scope-mri/tree/main/src) log information to TensorBoard, including learning rate, performance metrics, and the middle slice of the first MRI in the first batch for train/val/test per epoch (helps with inspecting augmentation and verifying data loading code)
 - To view TensorBoard logs, after activating your conda environment (with TensorBoard installed), do: ```tensorboard --logdir=/path/to/logdir/job_name --port 6006```
   - Replace ```'path/to/logdir/'``` with the actual path, and make sure to update it in [`labrum_train.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_train.py) and [`labrum_tune.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_tune.py)
   - Use the ```'job_name'``` from when you began training/tuning
   - Then, either access ```http://localhost:6006``` in your browser
   - Or if on an HPC, ssh into the computer with a new terminal tab ```ssh -L 6006:localhost:6006 myaccount@example_computer.edu```, then access ```http://localhost:6006``` in your browser
   - You can use a different port (6006 is chosen as an example)

## Citation

Please cite both papers associated with this repository and dataset:

```bibtex
@misc{sethi2025scopemribankartlesiondetection,
      title={SCOPE-MRI: Bankart Lesion Detection as a Case Study in Data Curation and Deep Learning for Challenging Diagnoses}, 
      author={Sahil Sethi and Sai Reddy and Mansi Sakarvadia and Jordan Serotte and Darlington Nwaudo and Nicholas Maassen and Lewis Shi},
      year={2025},
      eprint={2504.20405},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2504.20405}, 
}
```
```bibtex
@inproceedings{sethi_toward_2025,
	title = {Toward non-invasive diagnosis of {Bankart} lesions with deep learning},
	volume = {13407},
	url = {https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13407/134073G/Toward-non-invasive-diagnosis-of-Bankart-lesions-with-deep-learning/10.1117/12.3046251.full},
	doi = {10.1117/12.3046251},
	booktitle = {Medical {Imaging} 2025: {Computer}-{Aided} {Diagnosis}},
	publisher = {SPIE},
	author = {Sethi, Sahil and Reddy, Sai and Sakarvadia, Mansi and Serotte, Jordan and Nwaudo, Darlington and Maassen, Nicholas and Shi, Lewis},
	month = apr,
	year = {2025},
	pages = {854--863},
}
```

