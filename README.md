# scope-mri
This repository accompanies the release of the SCOPE-MRI dataset—a publicly available shoulder MRI dataset with image-level labels for several different pathologies. The dataset is described in our paper:

> [**SCOPE-MRI: Bankart Lesion Detection as a Case Study in Data Curation and Deep Learning for Challenging Diagnoses**](...)<br/>
  Sahil Sethi, Sai Reddy, Mansi Sakarvadia, Jordan Serotte, Darlington Nwaudo, Nicholas Maassen, & Lewis Shi. <b>arXiv</b>, preprint under review.

This is an extension of the work described in our prior paper:

> [**Toward Non-Invasive Diagnosis of Bankart Lesions with Deep Learning**](https://arxiv.org/abs/2412.06717)<br/>
  Sahil Sethi, Sai Reddy, Mansi Sakarvadia, Jordan Serotte, Darlington Nwaudo, Nicholas Maassen, & Lewis Shi. <b>Proc. SPIE 13407, Medical Imaging 2025: Computer-Aided Diagnosis.</b>, In Press at SPIE.  


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
The data used in our study has been released on the [`Medical Imaging and Data Resource Center (MIDRC)`](https://www.midrc.org/). Non-commercial access is freely available per MIDRC's usage policies to government and academic researchers. You can search for our MRIs in their system and download the DICOMs (~67 GB). Then, follow the data preprocessing steps below.

## Preprocessing
- [`MRI_and_metadata_import.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/MRI_and_metadata_import.py):
- [`train_test_val_creation.py`](https://github.com/sahilsethi0105/ortho_ml/blob/main/train_test_val_creation.py): 

## Training/CV, tuning, and testing
- [`labrum_train.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_train.py): trains models, does cross-validation, and does inference using either MRNet data or SCOPE-MRI
- [`labrum_tune.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_tune.py): tunes models using either MRNet data or SCOPE-MRI
- See the README in the src/ folder for specific commands

## Grad Cam: Interpreting what our models learned
- [`grad_cam_med.py`](https://github.com/sahilsethi0105/scope-mri/blob/grad_cam/grad_cam/grad_cam_med.py): outputs GradCam heat maps of what the model is "looking at" in each image
- See the README in the grad_cam/ folder for specific commands

## Additional Notes
 - The commands in the [`src/README.md`](https://github.com/sahilsethi0105/scope-mri/tree/main/src#readme) are for directly running the files
 - [`scripts/`](https://github.com/sahilsethi0105/scope-mri/tree/main/scripts) contains the shell scripts used to submit jobs to SLURM if using a HPC
 - The training/tuning files in [`src/`](https://github.com/sahilsethi0105/scope-mri/tree/main/src) log information to TensorBoard, including learning rate, performance metrics, and the middle slice of the first MRI in the first batch for train/val/test per epoch (helps with inspecting augmentation and verifying data loading code)

  To view TensorBoard logs, do:
  ```
  tensorboard --logdir=/path/to/logdir/job_name --port 6006
  ```
   - Then, either access: ```http://localhost:6006```
  Or if on an HPC, ssh into the computer with a new terminal tab: ```ssh -L 6006:localhost:6006 myacount@example_computer.edu```
   - Replace ```'path/to/logdir/'``` with the actual path, and make sure to update it in the training/tuning file, and use the 'job_name' from when you began training/tuning
   - You can use a different port (6006 is chosen as an example)
  
 - If using a HPC, you will want an easy way to view the GradCAM outputs, as running [`grad_cam_med.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/grad_cam/grad_cam_med.py) will produce heatmaps for every slice of every MRI in your test set; we found that the easiest way is to ssh into your computer with VSCode, then use the file explorer to click around and look at the slices

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

