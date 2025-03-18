## Update - Iterating through all images

Notice: 
- `use_label`: 1=use, 0=don't use, this will iterate through the middle slice of all MRI's in the preprocessed folder
- `all_slices`: 0=use middle slice, 1=iterate through all slices

`python grad_cam_med.py --use_label 1 --model_weights '/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/non_resized_MRIs/final_SLAP_data/cv_test7r2/cycle_5/cv_test7r2_model_cycle_5.pth' --model_type SwinTransformerV1 --label_column 'Anterior Inferior Labrum' --view coronal --fat_sat all --sequence_type all --dataset_type "labrum" --preprocessed_folder '/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/non_resized_MRIs/final_SLAP_data' --contrast_or_no "WO" --all_slices 1`

------------------
![image](https://github.com/user-attachments/assets/b28e8c26-9286-4dba-9859-b08a2fadc19c)

`python grad_cam_med.py --model_weights '/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/non_resized_MRIs/final_SLAP_data/ant_tuneV1r1/ant_tuneV1r1_trial_87_model.pth' --model_type VisionTransformer --label_column 'Anterior Inferior Labrum' --view sagittal --fat_sat all --sequence_type all --dataset_type "labrum" --preprocessed_folder '/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/non_resized_MRIs/final_SLAP_data'`

![image](https://github.com/user-attachments/assets/cff7e101-1ebf-49e1-84ac-a4a353d41bfb)

`python grad_cam_med.py --model_weights '/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/non_resized_MRIs/final_SLAP_data/cv_test6V/cycle_5/cv_test6V_model_cycle_5.pth' --model_type VisionTransformer --label_column 'Anterior Inferior Labrum' --view axial --fat_sat all --sequence_type all --dataset_type "labrum" --preprocessed_folder '/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/non_resized_MRIs/final_SLAP_data'`

![image](https://github.com/user-attachments/assets/994e0ffd-da74-49e7-a956-ac321a3881b9)

`python grad_cam_med.py --model_weights '/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/non_resized_MRIs/final_SLAP_data/cv_test7V/cycle_8/cv_test7V_model_cycle_8.pth' --model_type VisionTransformer --label_column 'Anterior Inferior Labrum' --view coronal --fat_sat all --sequence_type all --dataset_type "labrum" --preprocessed_folder '/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/non_resized_MRIs/final_SLAP_data'`

![image](https://github.com/user-attachments/assets/b30c8e46-cb3d-4bcd-b762-8b50b890148e)

`python grad_cam_med.py --model_weights '/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/non_resized_MRIs/final_SLAP_data/cv_ant1_33Vr3/cycle_5/cv_ant1_33Vr3_model_cycle_5.pth' --model_type VisionTransformer --label_column 'Anterior Inferior Labrum' --view sagittal --fat_sat all --sequence_type all --dataset_type "labrum" --preprocessed_folder '/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/non_resized_MRIs/final_SLAP_data'`

![image](https://github.com/user-attachments/assets/dd45b4c4-7a89-4aa2-974d-249ce96a2bb4)

`python grad_cam_med.py --model_weights '/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/non_resized_MRIs/final_SLAP_data/labrum_v1/labrum_v1_model_final.pth' --model_type AlexNet --label_column 'Anterior Inferior Labrum' --view sagittal --fat_sat all --sequence_type all --dataset_type "labrum" --preprocessed_folder '/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/non_resized_MRIs/final_SLAP_data' --contrast_or_no "all"`

![image](https://github.com/user-attachments/assets/a7b2cb5a-72a6-44e0-afa4-019d746593c3)

`python grad_cam_med.py --model_weights '/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/non_resized_MRIs/final_SLAP_data/cv_test7r2/cycle_5/cv_test7r2_model_cycle_5.pth' --model_type SwinTransformerV1 --label_column 'Anterior Inferior Labrum' --view coronal --fat_sat all --sequence_type all --dataset_type "labrum" --preprocessed_folder '/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/non_resized_MRIs/final_SLAP_data' --contrast_or_no "WO"`


How to get a node:
`srun --pty -p gpuq --cpus-per-task=4 --gres=gpu:1 /bin/bash`
