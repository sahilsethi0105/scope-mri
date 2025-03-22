# scope-mri
Deep learning for Bankart lesion detection

[`Link to Paper`](...)


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
Due to the conditions of our Institutional Review Board (IRB) and Data Use Agreement, data must be obtained in DICOM format directly from the repository. However, we have included our preprocessing code: 
- [`MRI_and_metadata_import.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/MRI_and_metadata_import.py): 

## Train
- [`labrum_train.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_train.py): outputs GradCam heat maps of what the model is "looking at" in each image

## Tune
- [`labrum_tune.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_tune.py): outputs GradCam heat maps of what the model is "looking at" in each image

## Grad Cam: Interpreting what our models learned
- [`grad_cam_med.py`](https://github.com/sahilsethi0105/scope-mri/blob/grad_cam/grad_cam/grad_cam_med.py): outputs GradCam heat maps of what the model is "looking at" in each image



