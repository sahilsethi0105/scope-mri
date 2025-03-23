# SCOPE-MRI: Intro to deep learning for MRIs/CTs
This repository is meant to be a useful resource for getting started with using deep learning on MRIs and CT scans. It accompanies the release of the Shoulder Comprehensive Orthopedic Pathology Evaluation (SCOPE)-MRI dataset—a publicly available shoulder MRI dataset with image-level labels for several different pathologies. The dataset is described in our paper:

> [**SCOPE-MRI: Bankart Lesion Detection as a Case Study in Data Curation and Deep Learning for Challenging Diagnoses**](...)<br/>
  Sahil Sethi, Sai Reddy, Mansi Sakarvadia, Jordan Serotte, Darlington Nwaudo, Nicholas Maassen, & Lewis Shi. <b>arXiv</b>, preprint under review.

This is an extension of the work described in our prior paper:

> [**Toward Non-Invasive Diagnosis of Bankart Lesions with Deep Learning**](https://arxiv.org/abs/2412.06717)<br/>
  Sahil Sethi, Sai Reddy, Mansi Sakarvadia, Jordan Serotte, Darlington Nwaudo, Nicholas Maassen, & Lewis Shi. <b>Proc. SPIE 13407, Medical Imaging 2025: Computer-Aided Diagnosis.</b>, In Press at SPIE.

Although the repo was developed for the above papers, we have written the code so that it can be easily adapted for training, hyperparameter tuning, cross-validating, and using GradCAM for any binary classification task using MRIs or CT scans. 
 - If you have your own dataset, [`MRI_and_metadata_import.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/MRI_and_metadata_import.py) and [`train_test_val_creation.py`](https://github.com/sahilsethi0105/ortho_ml/blob/main/train_test_val_creation.py) show how to go from the raw DICOM files to a final, preprocessed dataset
 - In ```loader.py```, create your custom dataset structure (include any desired additional preprocessing and/or augmentation code), and update ```prepare_and_create_loaders()``` and ```prepare_and_create_loaders_from_params()``` accordingly (they are used for ```labrum_train.py``` and ```labrum_tune.py```, respectively)
 - Many input arguments for ```labrum_train.py``` and ```labrum_tune.py``` are specific for our SCOPE-MRI dataset (called ```labrum``` in the ```dataset_type``` argument)
   - When set to ```MRNet```, many of these arguments are not used
   - You can remove and/or modify these based on your code for preparing dataloaders
 - You can add custom models or modify the existing ones in [`models.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/models.py)
 - This version of the codebase is designed to train all models on a single GPU, but manually implements much of the same behavior as PyTorch Lightning (eg, checkpointing, Optuna integration, TensorBoard logging); if you wish to train on multiple GPUs, we recommend using [`this article`](https://lightning.ai/docs/pytorch/stable/starter/converting.html) to adapt the code 
   - You will also need to change ```job_labrum_train.sh``` to request multiple GPUs on your HPC
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
If you simply want to use this repo to get familiar with deep learning for MRIs/CTs, we recommend installing the [`Stanford MRNet dataset available here`](https://stanfordmlgroup.github.io/competitions/mrnet/). Their dataset is larger than ours and significantly easier to train on. 

However, if you are interested in the SCOPE-MRI dataset, it has been released on the [`Medical Imaging and Data Resource Center (MIDRC)`](https://www.midrc.org/). Non-commercial access is freely available per MIDRC's usage policies to government and academic researchers. You can search for our MRIs in their system and download the DICOMs (~67 GB) _**UPDATE WITH EXACT INSTRUCTIONS**_. Then, follow the data preprocessing steps below. 

## Using the Repo with SCOPE-MRI
- [`MRI_and_metadata_import.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/MRI_and_metadata_import.py):
- [`train_test_val_creation.py`](https://github.com/sahilsethi0105/ortho_ml/blob/main/train_test_val_creation.py):

## Using the Repo with MRNet
 - F

## Visualizing MRIs
 - F

## Training, Cross-Validation, and Hyperparameter Tuning
- [`labrum_train.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_train.py): trains models, does cross-validation, and does inference using either MRNet data or SCOPE-MRI
- [`labrum_tune.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_tune.py): tunes models using either MRNet data or SCOPE-MRI
- See the README in the ```src/``` folder for specific Python commands

## Grad-CAM: Interpreting what a model learned
- [`grad_cam_med.py`](https://github.com/sahilsethi0105/scope-mri/blob/grad_cam/grad_cam/grad_cam_med.py): outputs Grad-CAM heat maps of what the model is "looking at" in each image (one heatmap for each slice in each MRI in the test set)
- See the README in the ```grad_cam/``` folder for specific Python commands

## Additional Notes
 - The commands in [`src/README.md`](https://github.com/sahilsethi0105/scope-mri/tree/main/src#readme) are for directly running the files
 - [`scripts/`](https://github.com/sahilsethi0105/scope-mri/tree/main/scripts) contains the shell scripts used to submit jobs to SLURM if using an HPC
 -  If using an HPC, you will want an easy way to view the GradCAM outputs because running [`grad_cam_med.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/grad_cam/grad_cam_med.py) will produce heatmaps for every slice of every MRI in your test set; we found that the easiest way is to ssh into your computer with VSCode, then use the file explorer to click around and look at the slices
 - The files in [`src/`](https://github.com/sahilsethi0105/scope-mri/tree/main/src) log information to TensorBoard, including learning rate, performance metrics, and the middle slice of the first MRI in the first batch for train/val/test per epoch (helps with inspecting augmentation and verifying data loading code)

  To view TensorBoard logs, after activating your conda environment (with TensorBoard installed), do:
  ```
  tensorboard --logdir=/path/to/logdir/job_name --port 6006
  ```
   - Replace ```'path/to/logdir/'``` with the actual path, and make sure to update it in ```labrum_train.py``` and ```labrum_tune.py ```
   - Use the ```'job_name'``` from when you began training/tuning
   - Then, either access ```http://localhost:6006``` in your browser
   - Or if on an HPC, ssh into the computer with a new terminal tab ```ssh -L 6006:localhost:6006 myaccount@example_computer.edu```, then access ```http://localhost:6006``` in your browser
   - You can use a different port (6006 is chosen as an example)

## Citation

Please cite both papers associated with this repository and dataset **(UPDATE with SCOPE-MRI citation after arXiv post)**:

```bibtex
@misc{sethi2024noninvasivediagnosisbankartlesions,
      title={Toward Non-Invasive Diagnosis of Bankart Lesions with Deep Learning}, 
      author={Sahil Sethi and Sai Reddy and Mansi Sakarvadia and Jordan Serotte and Darlington Nwaudo and Nicholas Maassen and Lewis Shi},
      year={2024},
      eprint={2412.06717},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.06717}, 
}
```
```bibtex
@misc{sethi2024noninvasivediagnosisbankartlesions,
      title={Toward Non-Invasive Diagnosis of Bankart Lesions with Deep Learning}, 
      author={Sahil Sethi and Sai Reddy and Mansi Sakarvadia and Jordan Serotte and Darlington Nwaudo and Nicholas Maassen and Lewis Shi},
      year={2024},
      eprint={2412.06717},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.06717}, 
}
```

