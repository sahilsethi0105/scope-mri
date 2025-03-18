import argparse
from models import CNN3D, ResNet50, AlexNet, VisionTransformer, SwinTransformerV1
from training_functions import train_model, evaluate_model, load_checkpoint, save_checkpoint, find_latest_checkpoint, compute_pos_weight
from data_loading import prepare_datasets, load_metadata
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
from monai.transforms import (
    Compose, RandFlipd, RandRotated
)
import matplotlib.pyplot as plt

# Function to convert string to boolean
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


preprocessed_folder = '/Users/sahilsethi/Desktop/Ortho ML Labrum Tears/preprocessed_MRIs/SLAP_from_all_labels'
label_column = 'SLAP Label'
batch_size = 1
num_epochs = 10
job_name = 'test_job_swinv1'
model_type = 'SwinTransformerV1'
lr = 1e-5
weight_decay = 0.01
dropout_rate = 0.3
augment = False
augment_factor = 1
#model_weights = '/gpfs/data/orthopedic-lab/MRNet-v1.0/MRNet-v1.0/trial5_alexnet/cnn3d_model_first50.pth'
model_weights = False #'/Users/sahilsethi/Desktop/Ortho ML Labrum Tears/MRNet-v1.0/cnn3d_model_final.pth'

# Prepare datasets
train_loader, val_loader, test_loader = prepare_datasets(batch_size=batch_size, preprocessed_folder=preprocessed_folder, label_column=label_column, augment=augment, augment_factor=augment_factor)


# Define augmentation transforms using MONAI
transform = Compose([
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),  # Vertical flip
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),  # Horizontal flip
    RandRotated(keys=["image"], prob=0.5, range_x=5 * 3.1415926535 / 180, range_y=0, range_z=0, mode="bilinear")  # Rotation in HxW plane
])

# Function to display a slice from the MRI volume
def display_slices(volume, x_slice, y_slice, z_slice, title_prefix=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display X-axis slice
    axes[0].imshow(volume[x_slice, :, :], cmap='gray')
    axes[0].set_title(f'{title_prefix} X-axis slice {x_slice}')
    
    # Display Y-axis slice
    axes[1].imshow(volume[:, y_slice, :], cmap='gray')
    axes[1].set_title(f'{title_prefix} Y-axis slice {y_slice}')
    
    # Display Z-axis slice
    axes[2].imshow(volume[:, :, z_slice], cmap='gray')
    axes[2].set_title(f'{title_prefix} Z-axis slice {z_slice}')
    
    plt.show()

# Specify the slice numbers to display
x_slice = 112
y_slice = 112
z_slice = 112

# Print the first batch of combined training data
for batch in train_loader:
    print(batch[0].shape, batch[1])
    first_mri = batch[0][0, 0].numpy()  # First MRI in the batch

    # Display original slices
    display_slices(first_mri, x_slice, y_slice, z_slice, title_prefix="Original")

    # Apply transformations
    transformed_mri = transform({"image": batch[0][0]})["image"].numpy()
    
    # Display transformed slices
    display_slices(transformed_mri[0], x_slice, y_slice, z_slice, title_prefix="Transformed")
    
    break
