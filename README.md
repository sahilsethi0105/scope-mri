# scope-mri
This repository accompanies the release of the SCOPE-MRI dataset—a publically available shoulder MRI dataset with image-level labels for several different pathologies. 

SCOPE-MRI: Bankart Lesion Detection as a Case Study in Data Curation and Deep Learning for Challenging Diagnoses. [`Link to Paper`](...)

Sethi, S., Reddy, S., Sakavadia, M., Serotte, J., Nwaudo, D., Maassen, N., & Shi, L. Toward Non-Invasive Diagnosis of Bankart Lesions with Deep Learning. Proc. SPIE 13407, Medical Imaging 2025: Computer-Aided Diagnosis. [`Link to Paper`](https://arxiv.org/abs/2412.06717)

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

Please cite this work as:

```bibtex
@inproceedings{sethi2025scope,
...
}
```
## Data Availability
The data used in our study has been released on: https://www.midrc.org/
Due to the conditions of our Institutional Review Board (IRB) and Data Use Agreement, data must be obtained in DICOM format directly from the repository. \ However, we have included our preprocessing code: 
- [`MRI_and_metadata_import.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/MRI_and_metadata_import.py): 

## Running training, tuning, and testing
- [`labrum_train.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_train.py): trains models, does cross-validation, and does inference using either MRNet data or SCOPE-MRI
- [`labrum_tune.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_tune.py): tunes models using either MRNet data or SCOPE-MRI
- See the README in the src/ folder for specific commands

## Grad Cam: Interpreting what our models learned
- [`grad_cam_med.py`](https://github.com/sahilsethi0105/scope-mri/blob/grad_cam/grad_cam/grad_cam_med.py): outputs GradCam heat maps of what the model is "looking at" in each image
- See the README in the grad_cam/ folder for specific commands



