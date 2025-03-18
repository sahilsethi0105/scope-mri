# scope-mri
Deep learning for Bankart lesion detection

[`Link to Paper`](...)

# Grad Cam: Interpreting what our models learned
- [`grad_cam_med.py`](https://github.com/sahilsethi0105/scope-mri/blob/grad_cam/grad_cam/grad_cam_med.py): outputs GradCam heat maps of what the model is "looking at" in each image


## Installation

Requirements:

- `python >=3.7,<3.11`

```bash
git clone https://github.com/msakarvadia/memorization.git
cd memorization
conda create -p env python==3.10
conda activate env
pip install -r requirements.txt
```

### Setting Up Pre-Commit Hooks (for nice code formatting)

#### Black

To maintain consistent formatting, we take advantage of `black` via pre-commit hooks.
There will need to be some user-side configuration. Namely, the following steps:

1. Install black via `pip install black` (included in `requirements.txt`).
2. Install `pre-commit` via `pip install pre-commit` (included in `requirements.txt`).
3. Run `pre-commit install` to setup the pre-commit hooks.

Once these steps are done, you just need to add files to be committed and pushed and the hook will reformat any Python file that does not meet Black's expectations and remove them from the commit. Just re-commit the changes and it'll be added to the commit before pushing.



## Citation

Please cite this work as:

```bibtex
@inproceedings{sethi2025scope,
...
}
```

