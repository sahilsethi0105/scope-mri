# scope-mri
This repository accompanies the release of the SCOPE-MRI dataset—a publicly available shoulder MRI dataset with image-level labels for several different pathologies. The dataset is described in our paper _SCOPE-MRI: Bankart Lesion Detection as a Case Study in Data Curation and Deep Learning for Challenging Diagnoses._ [`Link to Paper`](...)

This is an extension of the work described in our prior paper:

> [**Toward Non-Invasive Diagnosis of Bankart Lesions with Deep Learning**](https://arxiv.org/abs/2412.06717)<br/>
  Sahil Sethi, Sai Reddy, Mansi Sakarvadia, Jordan Serotte, Darlington Nwaudo, Nicholas Maassen, & Lewis Shi. <b>Proc. SPIE 13407, Medical Imaging 2025: Computer-Aided Diagnosis.>, In Press at SPIE.  


## Installation

Requirements:

- `python==3.10`

```bash
git clone https://github.com/sahilsethi0105/scope-mri.git
cd scope-mri
conda env create -f environment.yml
conda activate ortho_env 
```

## Citation

Please cite this work as **(UPDATE with SCOPE-MRI citation after arXiv post)**:

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
## Data Availability
The data used in our study has been released on the [`Medical Imaging and Data Resource Center (MIDRC)`](https://www.midrc.org/). Non-commercial access is freely available per MIDRC's usage policies to government and academic researchers. You can search for our MRIs in their system and download the DICOMs (~67 GB). Then, follow the data preprocessing steps below.

## Preprocessing
- [`MRI_and_metadata_import.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/MRI_and_metadata_import.py):
- [`train_test_val_creation.py`](https://github.com/sahilsethi0105/ortho_ml/blob/main/train_test_val_creation.py): 

## Running training, tuning, and testing
- [`labrum_train.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_train.py): trains models, does cross-validation, and does inference using either MRNet data or SCOPE-MRI
- [`labrum_tune.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_tune.py): tunes models using either MRNet data or SCOPE-MRI
- See the README in the src/ folder for specific commands

## Grad Cam: Interpreting what our models learned
- [`grad_cam_med.py`](https://github.com/sahilsethi0105/scope-mri/blob/grad_cam/grad_cam/grad_cam_med.py): outputs GradCam heat maps of what the model is "looking at" in each image
- See the README in the grad_cam/ folder for specific commands



