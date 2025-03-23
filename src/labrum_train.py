import argparse
from models import CNN3D, ResNet50, AlexNet, VisionTransformer, SwinTransformerV1, SwinTransformerV2, ResNet34, DenseNet, EfficientNet
from training_functions import train_model, evaluate_model, load_checkpoint, save_checkpoint, find_latest_checkpoint, compute_pos_weight, load_model_weights, cross_validate_model
from loader import prepare_and_create_loaders, load_metadata
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import io
import random
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
from torch.utils.tensorboard import SummaryWriter

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

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--preprocessed_folder', type=str, required=True) # Path to folder containing "train", "val", and "test" subfolders
parser.add_argument('--label_column', type=str, required=True) # Label column in metadata.csv for train, val, and test subfolders
parser.add_argument('--view', type=str, required=True, help='Selected view (axial, sagittal, coronal, ABERS, all)')
parser.add_argument('--batch_size', type=int, required=True) # This should be 1 for all 2D models, CNN3D must have > 1
parser.add_argument('--num_epochs', type=int, required=True)
parser.add_argument('--job_name', type=str, required=True) # Job name for saving results to subfolders
parser.add_argument('--model_type', type=str, required=True, help='Model type (ResNet50, MRNet, or CNN3D)')
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--weight_decay', type=float, required=True)
parser.add_argument('--dropout_rate', type=float, required=False)
parser.add_argument('--augment', type=str2bool, nargs='?', const=True, default=True, help="Whether to apply data augmentation during training") # Keep this True always
parser.add_argument('--augment_factor', type=int, default=10) # Augmentation factor for positive class (label = 1)
parser.add_argument('--augment_factor_0', type=int, default=10)  # Augmentation factor for negative class (label = 0)
parser.add_argument('--model_weights', type=str, default=None, help='Path to model weights to initialize from') # Use this argument to specify a model pre-trained on MRNet
parser.add_argument('--transform_val', type=str2bool, nargs='?', const=True, default=True, help="Whether to apply data augmentation during training") # Keep this True always
parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau', help='Scheduler type (ReduceLROnPlateau, CosineAnnealingLR)')
parser.add_argument('--sequence_type', type=str, default='all', help='Sequence type to include (e.g., T1, T2, all)') 
parser.add_argument('--fat_sat', type=str, default='all', help='Fat saturation to include (Yes, No, all)')
parser.add_argument('--contrast_or_no', type=str, default='all', help='Contrast type to include (WO, W, WWO, all)') # Note that we exclude WWO in our study
parser.add_argument('--dataset_type', type=str, required=True, choices=['labrum', 'MRNet'], help='Dataset type (labrum or MRNet)')
parser.add_argument('--pos_weight', type=str, required=True, help='Set pos_weight to "automatic" or an integer value.') # We always use 'automatic'
parser.add_argument('--script_mode', type=str, default='train', help='Script mode (train or CV)') # CV will initiate cross-validation using N cycles/folds
parser.add_argument('--ret_val_probs', type=str2bool, nargs='?', const=True, default=False, help="Whether to return inference probabilities on validation set")
parser.add_argument('--n_cycles', type=int, default=5, help='Number of folds for CV') 
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping') 
parser.add_argument('--save_checkpoints', type=str2bool, nargs='?', const=True, default=False, help="Whether to save weights each epoch for checkpointing") 
parser.add_argument('--seed', type=int, default=None, help='random seed for folds for CV') # Specify a random seed for reproducibility

args = parser.parse_args()

SEED = args.seed 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f'Running in script_mode: {args.script_mode}')

if (args.script_mode == 'train'): 
    trial_num = 0
    # Define the base log directory
    base_log_dir = f"/gpfs/data/orthopedic-lab/tb_logs/{args.job_name}"

    # Check if the log directory exists
    if os.path.exists(base_log_dir):
        # List all subdirectories in the base log directory
        subdirs = os.listdir(base_log_dir)

        # Filter out subdirectories that match the pattern 'trial_{number}'
        trial_dirs = [d for d in subdirs if re.match(r'trial_\d+', d)]

        if trial_dirs:
            # Extract the numbers from the existing trial directories
            trial_numbers = [int(re.search(r'\d+', d).group()) for d in trial_dirs]

            # Set trial_num to the next integer after the most recent trial
            trial_num = max(trial_numbers) + 1

    # Define the log directory for this trial
    log_dir = os.path.join(base_log_dir, f"trial_{trial_num}")
    writer = SummaryWriter(log_dir=log_dir, comment="")

    # Print the trial number for reference
    print(f"Logging to TensorBoard directory: {log_dir}")
    print(f"Current trial number: {trial_num}")


print(f"Using dataset type: {args.dataset_type}")
print("Label Column: ", args.label_column)

if args.dataset_type == 'labrum':
    # These arguments are only relevant for the labrum dataset
    sequence_type = args.sequence_type
    fat_sat = args.fat_sat
    contrast_or_no = args.contrast_or_no

    print("Sequence Type: ", sequence_type)
    print("FS: ", fat_sat)
    print("Contrast: ", contrast_or_no)
elif args.dataset_type == 'MRNet':
    # These arguments are irrelevant for the MRNet dataset
    sequence_type = None
    fat_sat = None
    contrast_or_no = None

if (args.script_mode == 'train'):
    # Prepare datasets
    
    if args.ret_val_probs: 
        combined_train_loader, combined_val_loader, combined_test_loader, combined_val_test_loader = prepare_and_create_loaders(args, num_workers=0)
        print("Returning val probs...")
    else: 
        combined_train_loader, combined_val_loader, combined_test_loader = prepare_and_create_loaders(args, num_workers=0)
        print("Not returning val probs...")

    print("Prepared dataloaders") 

    # Set device based on if running locally (MPS) or on HPC (CUDA)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Torch CUDA Device Count: ", torch.cuda.device_count())
        print("Torch CUDA Current Device: ", torch.cuda.current_device())
        print("Torch CUDA Device: ", torch.cuda.device(0))
        print("Torch CUDA Device Name: ", torch.cuda.get_device_name(0))
        print("Using CUDA device")

        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Model initialization
    if (args.model_type == 'ResNet50'):
        model = ResNet50(dropout_rate=args.dropout_rate).to(device)
    elif (args.model_type == 'AlexNet'):
        model = AlexNet(dropout_rate=args.dropout_rate).to(device)
    elif (args.model_type == 'CNN3D'):
        model = CNN3D(dropout_rate=args.dropout_rate).to(device)
    elif (args.model_type == 'VisionTransformer'):
        model = VisionTransformer(dropout_rate=args.dropout_rate).to(device)
    elif (args.model_type == 'SwinTransformerV1'):
        model = SwinTransformerV1(dropout_rate=args.dropout_rate).to(device)
    elif (args.model_type == 'SwinTransformerV2'):
        model = SwinTransformerV2(dropout_rate=args.dropout_rate).to(device)
    elif args.model_type == 'EfficientNet':
        model = EfficientNet(dropout_rate=args.dropout_rate).to(device)
    elif args.model_type == 'DenseNet':
        model = DenseNet(dropout_rate=args.dropout_rate).to(device)
    elif args.model_type == 'ResNet34':
        model = ResNet34(dropout_rate=args.dropout_rate).to(device)
    else:
        print("Unrecognized model type")

    # Print model type
    print("Model type:", f"{args.model_type}")

    # Calculate pos_weight based on input argument and dataset type
    if args.pos_weight.lower() == "automatic":
        if args.dataset_type == 'labrum':
            # For labrum dataset
            train_metadata_df = load_metadata('train', args.preprocessed_folder)
            pos_weight = compute_pos_weight(train_metadata_df, args.label_column, contrast_or_no=contrast_or_no, view=args.view).to(device)
            print('pos_weight calculated automatically for labrum:', pos_weight.item())
        elif args.dataset_type == 'MRNet':
            # For MRNet dataset
            train_metadata_df = pd.concat([
                pd.read_csv(os.path.join(args.preprocessed_folder, f'train-{args.label_column}.csv'), header=None, names=['mri_id', args.label_column]),
                pd.read_csv(os.path.join(args.preprocessed_folder, f'valid-{args.label_column}.csv'), header=None, names=['mri_id', args.label_column])
            ])
            pos_weight = compute_pos_weight(train_metadata_df, args.label_column).to(device)
            print('pos_weight calculated automatically for MRNet:', pos_weight.item())
    else:
        try:
            pos_weight_value = float(args.pos_weight)
            pos_weight = torch.tensor([pos_weight_value], device=device)
            print('pos_weight set manually:', pos_weight.item())
        except ValueError:
            raise ValueError("pos_weight should be 'automatic' or a numeric value.")

    augment_status = args.augment
    if augment_status: 
        print('Using data augmentation')
    else: 
        print('Not using data augmentation')

    if args.transform_val:
        print('Using transforms')
    else: 
        print('Not using transforms')

    # Initialize BCEWithLogitsLoss with pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Add a learning rate scheduler
    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        T_max = 0 #placeholder value to pass into train_model for hparam logging
    elif args.scheduler == 'CosineAnnealingLR':
        T_max = 10 #(args.num_epochs // 5))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max) 
    elif args.scheduler == 'CyclicLR':
        base_lr = args.lr * 0.1  # Set base_lr as 10% of initial learning rate
        max_lr = args.lr * 10    # Set max_lr as 10x the initial learning rate
        step_size_up = 2000      # Number of iterations to reach max_lr
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, mode='triangular')
        T_max = 0  # Placeholder since CyclicLR doesn't use T_max
    else:
        raise ValueError(f"Unrecognized scheduler type: {args.scheduler}")

    print("Scheduler: ", args.scheduler)

    # Checkpoint directory
    checkpoint_dir = os.path.join(args.preprocessed_folder, args.job_name, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Find the latest checkpoint
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)

    # Load model weights if provided, otherwise load from checkpoint if available
    if args.model_weights:
        if (args.num_epochs == 0): 
            start_epoch = 0
            best_acc = 0.0
            model = load_model_weights(model, args.model_weights, device, exclude_final=False)
        else: 
            start_epoch = 0
            best_acc = 0.0
            model = load_model_weights(model, args.model_weights, device, exclude_final=True)
    else:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            start_epoch, _, best_acc = load_checkpoint(latest_checkpoint, model, optimizer)
            print(f"Resuming training from epoch {start_epoch}; best validation accuracy so far is {best_acc}")
        else:
            start_epoch = 0
            best_acc = 0.0
            print("No checkpoint found, starting training from scratch.")

    # Define number of epochs for training
    num_epochs = args.num_epochs

    # Set CSV path based on dataset type (this saves results to a master CSV)
    if args.dataset_type == 'labrum':
        csv_path = "/gpfs/data/orthopedic-lab/ortho_ml/experiments/labrum_results.csv"
    elif args.dataset_type == 'MRNet':
        csv_path = "/gpfs/data/orthopedic-lab/ortho_ml/experiments/MRNet_results.csv"

    # Print the selected CSV path for confirmation
    print(f"Results CSV Path: {csv_path}")

    # Define hyperparameters to log
    hparams = {
        'job_name': args.job_name,
        'lr': args.lr,
        'label_column': args.label_column,
        'weight_decay': args.weight_decay,
        'dropout_rate': args.dropout_rate,
        'scheduler': args.scheduler,
        'T_max': T_max,
        'augment': args.augment,
        'augment_factor': args.augment_factor,
        'augment_factor_0': args.augment_factor_0,
        'transform_val': args.transform_val,
        'pos_weight': pos_weight.cpu().item(),
        'batch_size': args.batch_size,
        'model_type': args.model_type,
        'view': args.view,
        'sequence_type': args.sequence_type,
        'fat_sat': args.fat_sat,
        'contrast_or_no': args.contrast_or_no,
        'dataset_type': args.dataset_type
    }

    # Log hyperparameters to TensorBoard
    writer.add_hparams(hparams, {})

    if (num_epochs > 0): 
        model = train_model(model, 
                combined_train_loader, 
                combined_val_loader, 
                criterion, 
                optimizer, 
                scheduler, 
                num_epochs=num_epochs, 
                pos_weight=pos_weight,
                T_max=T_max,
                device=device, 
                save_path=checkpoint_dir, 
                start_epoch=start_epoch, 
                best_acc=best_acc, 
                save_checkpoints=args.save_checkpoints, #set to True to save model weights every epoch, and then be able to resume from checkpoint
                csv_path=csv_path, 
                args=args, 
                writer=writer,
                early_stopping_patience=args.patience
        )

        print("Model training complete")

        # Save the final model
        torch.save(model.state_dict(), os.path.join(args.preprocessed_folder, args.job_name, f'{args.job_name}_model_final.pth'))

    # Evaluate the final model on the validation set
    val_prob_csv_path = f'/gpfs/data/orthopedic-lab/ortho_ml/experiments/{args.job_name}_val_probs.csv'
    y_val_true, y_val_pred, y_val_prob = evaluate_model(model, combined_val_test_loader, criterion, device=device, csv_path=val_prob_csv_path, writer=writer)

    # Calculate val AUC
    val_auc = roc_auc_score(y_val_true, y_val_prob)

    writer.add_scalar('Val AUC', val_auc)
    print(f"Val AUC: {val_auc:.4f}")

    prob_csv_path = f'/gpfs/data/orthopedic-lab/ortho_ml/experiments/{args.job_name}_probs.csv'

    # Evaluate the model on the test set
    y_true, y_pred, y_prob = evaluate_model(model, combined_test_loader, criterion, device=device, csv_path=prob_csv_path, writer=writer) 

    # Classification report
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

    print("Model Training Complete")
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Sensitivity (Recall): {recall:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Calculate AUC
    auc = roc_auc_score(y_true, y_prob)
    print(f"AUC: {auc:.4f}")

    # Log the final metrics to TensorBoard
    writer.add_scalar('Test Acc', accuracy)
    writer.add_scalar('Test Recall', recall)
    writer.add_scalar('Test Specificity', specificity)
    writer.add_scalar('Test Precision', precision)
    writer.add_scalar('Test F1_Score', f1)
    writer.add_scalar('Test AUC', auc)

    # Log the confusion matrix as an image
    cm_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]  # Normalize

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.4)
    sns.heatmap(cm_normalized, annot=True, annot_kws={"size": 10}, cmap='Blues', fmt='.2f')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Log the confusion matrix as a figure
    writer.add_figure('Confusion Matrix', plt.gcf(), num_epochs)
    plt.close()

    # Plot ROC-AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')

    # Log the ROC-AUC plot as a figure
    writer.add_figure('ROC-AUC', plt.gcf(), num_epochs)
    plt.close()

    # Export ROC_AUC curve points to csv
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    d = {'thresholds': thresholds, 'tpr': tpr, 'fpr': fpr}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(args.preprocessed_folder, args.job_name, 'roc_points.csv'), index=False)
    
    # Close the TensorBoard writer
    writer.close()

    # Read existing CSV
    results_df = pd.read_csv(csv_path)

    # Find the row corresponding to this job
    job_row = results_df[(results_df['job_name'] == args.job_name) & (results_df['model_type'] == args.model_type)]

    if not job_row.empty:
        job_index = job_row.index[0]
        results_df.at[job_index, 'test_accuracy'] = accuracy
        results_df.at[job_index, 'test_sensitivity'] = recall
        results_df.at[job_index, 'test_specificity'] = specificity
        results_df.at[job_index, 'test_precision'] = precision
        results_df.at[job_index, 'test_f1_score'] = f1
        results_df.at[job_index, 'test_auc'] = auc
    else:
        new_row = {
            'job_name': args.job_name,
            'view': args.view,
            'model_type': args.model_type,
            'label_column': args.label_column,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'dropout_rate': args.dropout_rate,
            'augment': args.augment,
            'augment_factor': args.augment_factor,
            'augment_factor_0': args.augment_factor_0,
            'pos_weight': pos_weight.cpu().item(),
            'scheduler': args.scheduler,
            'T_max': T_max, 
            'transform_val': args.transform_val,
            'latest_epoch': None,
            'latest_epoch_train_loss': None,
            'latest_epoch_train_acc': None,
            'latest_epoch_val_loss': None,
            'latest_epoch_val_acc': None,
            'best_epoch': None,
            'best_epoch_train_loss': None,
            'best_epoch_train_acc': None,
            'best_epoch_val_loss': None,
            'best_epoch_val_acc': None,
            'test_accuracy': accuracy,
            'test_sensitivity': recall,
            'test_specificity': specificity,
            'test_precision': precision,
            'test_f1_score': f1,
            'test_auc': auc
        }
        results_df = results_df._append(new_row, ignore_index=True)

    # Save updated CSV
    results_df.to_csv(csv_path, index=False)
else:
    print(f'Beginning stratified {args.n_cycles}-fold cross-validation...')
    save_path = os.path.join(args.preprocessed_folder, args.job_name)

    # Ensure the save_path directory exists
    os.makedirs(save_path, exist_ok=True)

    cross_validate_model(save_path, args, n_cycles=args.n_cycles, save_checkpoints=False, num_workers=4, csv_path=None, early_stopping_patience=10)

    print(f'Cross-validation complete. Please separately evaluate final models from cycle folders at: {save_path}')
