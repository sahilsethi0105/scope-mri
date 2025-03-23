## Grad-CAM: visualizing what the model learned

Example command line: 
```bash
python3 grad_cam_med.py \
  --model_weights 'path/to/trained/model.pth' \
  --all_slices 1 \
  --use_label 1 \
  --preprocessed_folder '/path/to/directory/with/train_val_test_subdirectories/' \
  --label_column 'abnormal' \
  --model_type AlexNet \
  --view axial \
  --sequence_type 'all' \
  --fat_sat 'all' \
  --contrast_or_no 'WO' \
  --dataset_type "MRNet" \
```
Notes: 
 - ```model_weights```: specify the path to the weights for the model you want to use
 - ```use_label```: 1=calculate gradients with respect to label, 0=do not (keep at 1 for default)
 - ```all_slices```: 0=use middle slice of each MRI, 1=iterate through all slices of each MRI
 - The rest of the input arguments should be consistent with what you used in ```labrum_train.py``` or ```labrum_tune.py``` to train the model
 - If using an HPC, you will want an easy way to view the GradCAM outputs because running [`grad_cam_med.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/grad_cam/grad_cam_med.py) will produce heatmaps for every slice of every MRI in your test set; we found that the easiest way is to ssh into your computer with VSCode, then use the file explorer to click around and look at the slices
