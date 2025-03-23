# scope-mri
This repository is meant to be a useful resource for getting started with using deep learning on MRIs and CT scans. It accompanies the release of the SCOPE-MRI dataset—a publicly available shoulder MRI dataset with image-level labels for several different pathologies. The dataset is described in our paper:

> [**SCOPE-MRI: Bankart Lesion Detection as a Case Study in Data Curation and Deep Learning for Challenging Diagnoses**](...)<br/>
  Sahil Sethi, Sai Reddy, Mansi Sakarvadia, Jordan Serotte, Darlington Nwaudo, Nicholas Maassen, & Lewis Shi. <b>arXiv</b>, preprint under review.

This is an extension of the work described in our prior paper:

> [**Toward Non-Invasive Diagnosis of Bankart Lesions with Deep Learning**](https://arxiv.org/abs/2412.06717)<br/>
  Sahil Sethi, Sai Reddy, Mansi Sakarvadia, Jordan Serotte, Darlington Nwaudo, Nicholas Maassen, & Lewis Shi. <b>Proc. SPIE 13407, Medical Imaging 2025: Computer-Aided Diagnosis.</b>, In Press at SPIE.

Although the repo was developed for the above papers, we have written the code so that it can be easily adapted for training, hyperparameter tuning, cross-validating, and using GradCAM for any binary classification task using MRIs or CT scans. 
 - In ```loader.py```, create your custom dataset structure (include any desired preprocessing and/or augmentation code), and update ```prepare_and_create_loaders()``` and ```prepare_and_create_loaders_from_params()``` accordingly (they are used for ```labrum_train.py``` and ```labrum_tune.py```, respectively)
 - Many of input arguments for ```labrum_train.py``` and ```labrum_tune.py``` are specific for our SCOPE-MRI dataset (called ```labrum``` in the ```dataset_type``` argument)
   - When set to ```MRNet```, many of these arguments are not used
   - You can remove and/or modify these based on your code for preparing dataloaders
 - You can add custom models or modify the existing ones in [`models.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/models.py)
 - By default, the weights from the epoch with the highest validation accuracy are preserved for inference across training, CV, and tuning; early stopping patien
 - If you want to initialize a model with pre-trained weights, just pass the path to the model weights into the ```model_weights``` argument 
 - If you just want to do inference, pass the path to trained model weights into the ```model_weights``` argument and set ```num_epochs``` to 0
 - Across the whole repo, the default behavior is for the weights from the epoch with the highest validation accuracy to be retained for inference; these weights are saved regardless of if you have checkpointing on or not
 - Results will be automatically saved to CSV files and TensorBoard logs; exact predictions/logits and labels for each MRI are saved to ```{job_name}_probs.csv``` and ```{job_name}_val_probs.csv``` files to facilitate pushing to GitHub from an HPC, then pulling locally and creating figures in a Jupyter notebook
 - If you set the ```save_checkpoints``` argument in ```labrum_train.py``` to True, then weights will be saved each epoch and the code can resume from the latest checkpoint if training is interrupted (but beware of the storage cost; it is best to remove all the checkpoints after successfully training because they are no longer needed due to the highest val acc epoch weights being saved separately)
 - For cross-validation, the code will automatically handle resuming from the start of the latest fold/cycle if interrupted, but only if you pass a random seed into the ```seed``` argument (to ensure consistant splitting)
 - For hyperparameter tuning, model weights are only saved if they achieve above a certain threshold at inference (default is AUC > 0.70); for checkpointing, the Optuna study itself is saved and automatically resumed if you run the code again after being interrupted (but the model weights are not saved by default)
 - For multi-class classification, you will need to update the loss functions accordingly, as well as possibly modify some of the logging

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

However, if you are interested in the SCOPE-MRI dataset, it has been released on the [`Medical Imaging and Data Resource Center (MIDRC)`](https://www.midrc.org/). Non-commercial access is freely available per MIDRC's usage policies to government and academic researchers. You can search for our MRIs in their system and download the DICOMs (~67 GB). Then, follow the data preprocessing steps below. 

## Preprocessing
- [`MRI_and_metadata_import.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/MRI_and_metadata_import.py):
- [`train_test_val_creation.py`](https://github.com/sahilsethi0105/ortho_ml/blob/main/train_test_val_creation.py):

## Visualizing MRIs
 - F

## Training/CV, tuning, and testing
- [`labrum_train.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_train.py): trains models, does cross-validation, and does inference using either MRNet data or SCOPE-MRI
- [`labrum_tune.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_tune.py): tunes models using either MRNet data or SCOPE-MRI
- See the README in the ```src/``` folder for specific commands

## Grad Cam: Interpreting what our models learned
- [`grad_cam_med.py`](https://github.com/sahilsethi0105/scope-mri/blob/grad_cam/grad_cam/grad_cam_med.py): outputs GradCam heat maps of what the model is "looking at" in each image
- See the README in the ```grad_cam/``` folder for specific commands

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

