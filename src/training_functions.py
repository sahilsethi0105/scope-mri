import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from glob import glob
import torchvision
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
import optuna

import random
SEED = 42  # Or any fixed seed value
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Define a global worker initialization function
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Create a global generator for consistency
g = torch.Generator()
g.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def log_image(writer, tag, images, epoch):
    grid = torchvision.utils.make_grid(images)
    writer.add_image(tag, grid, epoch)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_checkpoint(model, optimizer, epoch, val_acc, best_acc, checkpoint_path):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc,
        'best_acc': best_acc
    }
    torch.save(state, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    return state['epoch'], state['val_acc'], state['best_acc']

def load_model_weights(model, checkpoint_path, device='cpu', exclude_final=False):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint structures
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:  # PyTorch Lightning checkpoint
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:  # Custom checkpoint
            state_dict = checkpoint['model']
        else:  # Assume it is a plain model state dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if exclude_final:
        # Print the original state_dict keys
        print("Original keys in state_dict:")
        for key in state_dict.keys():
            print(key)

        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
        
        # Print the modified state_dict keys
        print("\nKeys in state_dict after removing classifier:" if exclude_final else "\nKeys in state_dict without modification:")
        for key in state_dict.keys():
            print(key)

    # Load the state dict
    model.load_state_dict(state_dict, strict=False)

    print(f"Model weights loaded from {checkpoint_path}")
    return model

def compute_pos_weight(metadata_df, label_column, contrast_or_no=None, view=None):
    #label_counts = metadata_df[label_column].value_counts()
    #neg_count = label_counts.get(0, 0)  # Ensure we get the count for label 0
    #pos_count = label_counts.get(1, 0)  # Ensure we get the count for label 1
    #pos_weight = neg_count / pos_count
    #return torch.tensor([pos_weight], dtype=torch.float32)
        # Filter based on contrast_or_no if provided
    if contrast_or_no is not None:
        metadata_df = metadata_df[metadata_df['contrast_or_no'] == contrast_or_no]
    
    # Filter based on view if provided
    if view is not None:
        metadata_df = metadata_df[metadata_df['view'] == view]
    
    # Compute label counts
    label_counts = metadata_df[label_column].value_counts()
    neg_count = label_counts.get(0, 0)  # Ensure we get the count for label 0
    pos_count = label_counts.get(1, 0)  # Ensure we get the count for label 1

    # Handle cases where there are no positive or negative samples
    if pos_count == 0:
        raise ValueError("No positive samples found after filtering. Cannot compute pos_weight.")
    if neg_count == 0:
        raise ValueError("No negative samples found after filtering. Cannot compute pos_weight.")

    # Compute positive weight
    pos_weight = neg_count / pos_count
    return torch.tensor([pos_weight], dtype=torch.float32)

def update_results_csv(csv_path, results):
    # Define the columns and their data types
    columns = {
        "job_name": str,
        "view": str,
        "model_type": str,
        "label_column": str,
        "batch_size": int,
        "num_epochs": int,
        "lr": float,
        "weight_decay": float,
        "dropout_rate": float,
        "augment": bool,
        "augment_factor": int,
        "transform_val": bool,
        "latest_epoch": int,
        "latest_epoch_train_loss": float,
        "latest_epoch_train_acc": float,
        "latest_epoch_val_loss": float,
        "latest_epoch_val_acc": float,
        "best_epoch": int,
        "best_epoch_train_loss": float,
        "best_epoch_train_acc": float,
        "best_epoch_val_loss": float,
        "best_epoch_val_acc": float,
        "scheduler": str,
        "test_accuracy": float,
        "test_sensitivity": float,
        "test_specificity": float,
        "test_precision": float,
        "test_f1_score": float,
        "test_auc": float,
    }

    # Ensure the CSV file exists and has the correct structure
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        # Create an empty DataFrame with the defined columns
        df = pd.DataFrame(columns=columns.keys())
        df = df.astype(columns)  # Set the correct data types
        df.to_csv(csv_path, index=False)
        print(f"CSV file initialized at: {csv_path}")

    # Load the existing CSV or start fresh if it was just initialized
    df = pd.read_csv(csv_path)

    # Append the new results to the DataFrame
    df = df._append(results, ignore_index=True)
    df.to_csv(csv_path, index=False)

#Train model function with class weights
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=5, device='cpu', save_path=None, start_epoch=0, best_acc=0.0, pos_weight=None, T_max=0, save_checkpoints=True, csv_path=None, args=None, trial=None, return_acc=False, writer=None, early_stopping_patience=10):
    metrics = {'Epoch': [], 'Training Loss': [], 'Training Accuracy': [], 'Validation Loss': [], 'Validation Accuracy': [], 'Training AUC': [], 'Validation AUC': []}
    metrics_path = os.path.join(save_path, 'train_metrics.csv')

    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        metrics = metrics_df.to_dict('list')
    else:
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(metrics_path, index=False)

    best_model_wts = model.state_dict()  # Initialize best_model_wts with current model state
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epochs"):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        y_true_train, y_prob_train = [], []

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training Batches", leave=False)):
        #for inputs, labels in tqdm(train_loader, desc="Training Batches", leave=False):
        
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs) #No sigmoid yet because criterion has built in sigmoid if using BCEWithLogitsLoss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = torch.round(torch.sigmoid(outputs))  # Apply sigmoid before rounding; could change threshold, but is 0.5 here by default

            y_true_train.extend(labels.cpu().numpy())
            y_prob_train.extend(torch.sigmoid(outputs).detach().cpu().numpy())

            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            # Log a sample slice of training images
            if writer and (batch_idx == 0):  # Log only the first batch in each epoch
                slice_idx = inputs.shape[2] // 2  # Select the middle slice if 3D, otherwise use channel dimension
                log_image(writer, 'train_images', inputs[:, :, slice_idx, :, :], epoch)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train

        # Calculate AUC for the training set
        train_auc = roc_auc_score(np.concatenate(y_true_train), np.concatenate(y_prob_train)) if len(set(np.concatenate(y_true_train))) > 1 else 0.0

        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        y_true_val, y_prob_val = [], []

        with torch.no_grad():

            for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader, desc="Validation Batches", leave=False)):
            #for inputs, labels in tqdm(val_loader, desc="Validation Batches", leave=False):
            
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                outputs = model(inputs) #No sigmoid yet because criterion has built in sigmoid if using BCEWithLogitsLoss
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.round(torch.sigmoid(outputs))  # Apply sigmoid before rounding; could change threshold, but is 0.5 here by default
        
                y_true_val.extend(labels.cpu().numpy())
                y_prob_val.extend(torch.sigmoid(outputs).detach().cpu().numpy())

                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                # Log a sample slice of validation images
                if writer and (batch_idx == 0):  # Log only the first batch in each epoch
                    slice_idx = inputs.shape[2] // 2  # Select the middle slice if 3D, otherwise use channel dimension
                    log_image(writer, 'val_images', inputs[:, :, slice_idx, :, :], epoch)

        val_loss /= len(val_loader.dataset)
        val_acc = correct_val / total_val

        # Calculate AUC for the validation set
        val_auc = roc_auc_score(np.concatenate(y_true_val), np.concatenate(y_prob_val)) if len(set(np.concatenate(y_true_val))) > 1 else 0.0

        # Save metrics
        metrics['Epoch'].append(epoch + 1)
        metrics['Training Loss'].append(epoch_loss)
        metrics['Training Accuracy'].append(train_acc)
        metrics['Validation Loss'].append(val_loss)
        metrics['Validation Accuracy'].append(val_acc)
        metrics['Training AUC'].append(train_auc)
        metrics['Validation AUC'].append(val_auc)
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(metrics_path, index=False)

        # Log metrics to TensorBoard
        if writer:
            writer.add_scalar('Train Loss', epoch_loss, epoch + 1)
            writer.add_scalar('Train Acc', train_acc, epoch + 1)
            writer.add_scalar('Train AUC', train_auc, epoch + 1)
            writer.add_scalar('Val Loss', val_loss, epoch + 1)
            writer.add_scalar('Val Acc', val_acc, epoch + 1)
            writer.add_scalar('Val AUC', val_auc, epoch + 1)

            # Log learning rate
            current_lr = get_lr(optimizer)
            writer.add_scalar('Learning_Rate', current_lr, epoch + 1)

            # Log model weights histograms
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch + 1)

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val AUC: {val_auc:.4f}')

        scheduler_type = trial.params['scheduler'] if (trial is not None and 'scheduler' in trial.params) else args.scheduler

        # Report to Optuna trial (if trial is provided)
        if trial:
            trial.report(val_auc, epoch)

            # Check if the trial should be pruned
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if (scheduler_type == 'ReduceLROnPlateau'): 
            scheduler.step(val_loss)
        else: 
            scheduler.step()

        if (val_acc > best_acc) and (epoch > 1):
            best_acc = val_acc
            best_epoch = epoch + 1
            best_model_wts = model.state_dict()
            epochs_no_improve = 0
        else: 
            epochs_no_improve += 1

        if save_checkpoints:
            checkpoint_epoch_path = os.path.join(save_path, f'cnn3d_model_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch + 1, val_acc, best_acc, checkpoint_epoch_path)
    
        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch + 1}. No improvement for {early_stopping_patience} consecutive epochs.')
            break

    #Set best_epoch to correct value if it is still at 0
    if (best_epoch == 0) and (best_acc != 0):
        best_epoch = metrics['Epoch'][metrics['Validation Accuracy'].index(best_acc)] if metrics['Validation Accuracy'] else 0
        
        # Load the model weights corresponding to the best_epoch
        best_epoch_checkpoint = os.path.join(save_path, f'cnn3d_model_epoch_{best_epoch}.pth')
        if os.path.exists(best_epoch_checkpoint):
            best_model_wts = torch.load(best_epoch_checkpoint)['model']

    print(f'Best Val Accuracy (at epoch {best_epoch}): {best_acc:.4f}')
    model.load_state_dict(best_model_wts)

    # Log hyperparameters to TensorBoard for hyperparameter tuning
    if trial and writer:
        hparams = {
            'lr': trial.params['lr'] if (trial is not None and 'lr' in trial.params) else args.lr,
            'label_column': args.label_column,
            'weight_decay': trial.params['weight_decay'] if (trial is not None and 'weight_decay' in trial.params) else args.weight_decay,
            'dropout_rate': trial.params['dropout_rate'] if (trial is not None and 'dropout_rate' in trial.params) else args.dropout_rate,
            'scheduler': trial.params['scheduler'] if (trial is not None and 'scheduler' in trial.params) else args.scheduler,
            'T_max': T_max, 
            'augment': args.augment,
            'augment_factor': trial.params['augment_factor'] if (trial is not None and 'augment_factor' in trial.params) else args.augment_factor,
            'augment_factor_0': trial.params['augment_factor_0'] if (trial is not None and 'augment_factor_0' in trial.params) else args.augment_factor_0,
            'transform_val': args.transform_val,
            'pos_weight': pos_weight.cpu().item(),  
            'batch_size': args.batch_size,
            'model_type': args.model_type,
            'view': args.view,
            'sequence_type': args.sequence_type,
            'fat_sat': args.fat_sat,
            'contrast_or_no': args.contrast_or_no
        }
        final_metrics = {
            'hparam/Best_Val_Acc': best_acc,
            'hparam/Best_Val_AUC': max(metrics['Validation AUC']),
            'hparam/Best_Val_Loss': min(metrics['Validation Loss']),
            'hparam/Final_Val_Acc': val_acc,
            'hparam/Final_Val_AUC': val_auc,
            'hparam/Final_Val_Loss': val_loss
        }
        writer.add_hparams(hparams, final_metrics)

    # Update results CSV
    if csv_path and args:
        results = {
            'job_name': args.job_name,
            'view': getattr(args, 'view', getattr(args, 'selected_train_view', None)),
            'model_type': args.model_type,
            'label_column': getattr(args, 'label_column', getattr(args, 'target_label', None)),
            'batch_size': args.batch_size, #trial.params['batch_size'] if (trial is not None and 'batch_size' in trial.params) else args.batch_size,
            'num_epochs': args.num_epochs,
            'lr': trial.params['lr'] if (trial is not None and 'lr' in trial.params) else args.lr,
            'weight_decay': trial.params['weight_decay'] if (trial is not None and 'weight_decay' in trial.params) else args.weight_decay,
            'dropout_rate': trial.params['dropout_rate'] if (trial is not None and 'dropout_rate' in trial.params) else args.dropout_rate,
            'pos_weight': pos_weight.cpu().item(), 
            'augment': args.augment,
            'augment_factor': trial.params['augment_factor'] if (trial is not None and 'augment_factor' in trial.params) else args.augment_factor,
            'augment_factor_0': trial.params['augment_factor_0'] if (trial is not None and 'augment_factor_0' in trial.params) else args.augment_factor_0,
            'scheduler': trial.params['scheduler'] if (trial is not None and 'scheduler' in trial.params) else args.scheduler,
            'T_max': T_max,
            'transform_val': args.transform_val,
            'latest_epoch': metrics['Epoch'][-1] if metrics['Epoch'] else 0,
            'latest_epoch_train_loss': metrics['Training Loss'][-1] if metrics['Training Loss'] else 0,
            'latest_epoch_train_acc': metrics['Training Accuracy'][-1] if metrics['Training Accuracy'] else 0,
            'latest_epoch_val_loss': metrics['Validation Loss'][-1] if metrics['Validation Loss'] else 0,
            'latest_epoch_val_acc': metrics['Validation Accuracy'][-1] if metrics['Validation Accuracy'] else 0,
            'best_epoch': best_epoch,
            'best_epoch_train_loss': metrics['Training Loss'][best_epoch - 1],
            'best_epoch_train_acc': metrics['Training Accuracy'][best_epoch - 1],
            'best_epoch_val_loss': metrics['Validation Loss'][best_epoch - 1],
            'best_epoch_val_acc': best_acc,
            'test_accuracy': 0,  # Placeholder, as test evaluation is done separately
            'test_sensitivity': 0,
            'test_specificity': 0,
            'test_precision': 0,
            'test_f1_score': 0,
            'test_auc': 0
        }
        update_results_csv(csv_path, results)
    if return_acc:
        return model, best_acc
    return model

def evaluate_model(model, test_loader, criterion, device='cpu', writer=None, csv_path='/gpfs/data/orthopedic-lab/ortho_ml/experiments/labrum_probs.csv', num_epochs=0):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    unique_ids, mri_ids, raw_outputs, probabilities, labels_list = [], [], [], [], []
    save_csv = False

    with torch.no_grad():        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing Batches")):
            if len(batch) == 4:
                inputs, labels, unique_id, mri_id = batch
                save_csv = True
            else:
                inputs, labels = batch
                unique_id = ["N/A"] * len(inputs)
                mri_id = ["N/A"] * len(inputs)
        
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            
            # Debugging: print raw outputs
            #print(f"Batch {batch_idx + 1} - MRI ID: {mri_id} - Unique ID: {unique_id} - Raw Outputs: {outputs.cpu().numpy()}")

            probs = torch.sigmoid(outputs) #apply sigmoid since it is not in the model architecture due to using BCEWithLogitsLoss
            preds = torch.round(probs) #round predictions to nearest integer (threshold = 0.5 here); could change threshold
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

            # Collect data for CSV
            unique_ids.extend(unique_id)
            mri_ids.extend(mri_id)
            raw_outputs.extend(outputs.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

            # Print probabilities
            #print(f"Batch {batch_idx + 1} - MRI ID: {mri_id} - Unique ID: {unique_id} - Probabilities: {probs.cpu().numpy()}")

            # Log a sample slice of testing images
            if writer and (batch_idx == 0):  # Log only the first batch
                slice_idx = inputs.shape[2] // 2  # Select the middle slice if 3D, otherwise use channel dimension
                log_image(writer, 'test_images', inputs[:, :, slice_idx, :, :], num_epochs)
            
    # Save to CSV
    if save_csv:
        results_df = pd.DataFrame({
            'mri_id': mri_ids,
            'unique_id': unique_ids,
            'raw_output': [item[0] for item in raw_outputs],  # flatten nested lists
            'probability': [item[0] for item in probabilities],  # flatten nested lists
            'label': [item[0] for item in labels_list]  # flatten nested lists
        })

        results_df.to_csv(csv_path, index=False)
        print(f"Probabilities saved to: {csv_path}")

    return y_true, y_pred, y_prob

def find_latest_checkpoint(checkpoint_dir):
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint

###Cross-validation functions
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from models import CNN3D, ResNet50, AlexNet, VisionTransformer, SwinTransformerV1, SwinTransformerV2, ResNet34, DenseNet, EfficientNet
from loader import load_metadata, MRIDataset3D, MRIDataset2D, MRNetDataset3D, MRNetDataset2D, load_labels
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold

def find_latest_cycle_folder(base_path):
    cycle_folders = sorted([f for f in os.listdir(base_path) if f.startswith("cycle_")], key=lambda x: int(x.split("_")[1]))
    if not cycle_folders:
        return None
    return os.path.join(base_path, cycle_folders[-1])


def find_latest_checkpoint_in_cycle(cycle_path):
    checkpoints = glob(os.path.join(cycle_path, "*.pth"))
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint


def load_combined_metadata(base_folder):
    train_metadata = load_metadata('train', base_folder)
    val_metadata = load_metadata('val', base_folder)
    
    train_metadata['subfolder'] = 'train'
    val_metadata['subfolder'] = 'val'
    
    combined_metadata = pd.concat([train_metadata, val_metadata], ignore_index=True)
    return combined_metadata

def cross_validate_model(save_path, args, n_cycles=5, save_checkpoints=True, num_workers=4, csv_path=None, early_stopping_patience=10):

    # Load and combine train and validation metadata
    if (args.dataset_type == 'labrum'): 
        #Labrum dataset label loading
        train_metadata_df = load_metadata('train', args.preprocessed_folder)
        val_metadata_df = load_metadata('val', args.preprocessed_folder)
        combined_metadata_df = pd.concat([train_metadata_df, val_metadata_df], ignore_index=True)

        test_df = load_metadata('test', args.preprocessed_folder) 
        #print(f"Test set dataframe size: {test_df.shape}")
        #print(test_df.head())

    else: 
        #MRNet dataset label loading
        combined_metadata_df, test_df = load_labels(args.preprocessed_folder, args.label_column)

    # Apply filtering based on view, sequence_type, fat_sat, and contrast_or_no
    if (args.dataset_type == 'labrum'): 
        #if args.view != 'all':
        #    combined_metadata_df = combined_metadata_df[combined_metadata_df['view'] == args.view]
        if args.sequence_type != 'all':
            combined_metadata_df = combined_metadata_df[combined_metadata_df['sequence_type'] == args.sequence_type]
        if args.fat_sat != 'all':
            combined_metadata_df = combined_metadata_df[combined_metadata_df['fat_sat'] == args.fat_sat]
        if args.contrast_or_no != 'all':
            combined_metadata_df = combined_metadata_df[combined_metadata_df['contrast_or_no'] == args.contrast_or_no]

    # Extract labels for stratified splitting
    mri_ids = combined_metadata_df['mri_id'].unique()
    labels = combined_metadata_df.groupby('mri_id')[args.label_column].first().values

    # Initialize StratifiedKFold (or rather StratifiedKCycle)
    skf = StratifiedKFold(n_splits=n_cycles, shuffle=True, random_state=args.seed)
    print(f"Splits made with seed: {args.seed}")

    cycle_results = []

    latest_cycle_folder = find_latest_cycle_folder(save_path) 
    latest_checkpoint = None
    start_cycle = 0  # Default to starting from the first cycle

    if latest_cycle_folder:
        latest_checkpoint = find_latest_checkpoint_in_cycle(latest_cycle_folder)
        if latest_checkpoint:
            start_cycle = (int(latest_cycle_folder.split("_")[-1]) - 1)
            print(f"Resuming from cycle {start_cycle + 1} using checkpoint {latest_checkpoint}")
        else:
            start_cycle = (int(latest_cycle_folder.split("_")[-1]) - 1)
            print(f"Resuming from cycle {start_cycle + 1} but no checkpoint found. Starting this cycle from scratch.")
    else:
        start_cycle = 0
        print("No existing cycle folders found. Starting from cycle 1.")

    # Iterate over each cycle
    for cycle, (train_idx, val_idx) in enumerate(skf.split(mri_ids, labels)):

        if cycle < start_cycle:
            continue  # Skip completed cycles

        # Set device based on if running locally (MPS) or on HPC (CUDA)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Torch CUDA Device Count: ", torch.cuda.device_count())
            print("Torch CUDA Current Device: ", torch.cuda.current_device())
            print("Torch CUDA Device: ", torch.cuda.device(0))
            print("Torch CUDA Device Name: ", torch.cuda.get_device_name(0))
            print("Using CUDA device")
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

        # Calculate pos_weight based on input argument and dataset type
        if args.pos_weight.lower() == "automatic":
            if args.dataset_type == 'labrum':
                # For labrum dataset
                train_metadata_df = load_metadata('train', args.preprocessed_folder)
                pos_weight = compute_pos_weight(train_metadata_df, args.label_column, contrast_or_no=args.contrast_or_no, view=args.view).to(device)
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

        # Initialize BCEWithLogitsLoss with pos_weight
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Add a learning rate scheduler
        if args.scheduler == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            T_max = 0 #placeholder value to pass into train_model for hparam logging
        elif args.scheduler == 'CosineAnnealingLR':
            T_max = 10
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        elif args.scheduler == 'CyclicLR':
            base_lr = args.lr * 0.1  # Set base_lr as 10% of initial learning rate
            max_lr = args.lr * 10    # Set max_lr as 10x the initial learning rate
            step_size_up = 2000  # Number of iterations to reach max_lr
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, mode='triangular')
            T_max = 0  # Placeholder since CyclicLR doesn't use T_max
        else:
            raise ValueError(f"Unrecognized scheduler type: {args.scheduler}")

        print("Scheduler: ", args.scheduler)

        # Load model weights if provided, otherwise load from checkpoint if available
        if cycle == start_cycle and latest_checkpoint:
            # Resume from the latest checkpoint
            start_epoch, optimizer_state, best_acc = load_checkpoint(latest_checkpoint, model, optimizer)
            print(f"Resuming training for cycle {cycle + 1} from epoch {start_epoch}, best validation accuracy: {best_acc}")
        else:
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
                start_epoch = 0
                best_acc = 0.0
                print("No weights found, starting training from scratch.")

        print(f"Cycle {cycle + 1}/{n_cycles}")
        train_mri_ids = mri_ids[train_idx]
        val_mri_ids = mri_ids[val_idx]

        # Split the combined metadata into training and validation sets for this cycle
        #train_df = combined_metadata_df[combined_metadata_df['mri_id'].isin(train_mri_ids)].copy()
        #val_df = combined_metadata_df[combined_metadata_df['mri_id'].isin(val_mri_ids)].copy()

        train_df = combined_metadata_df[
            (combined_metadata_df['mri_id'].isin(train_mri_ids)) &
            (combined_metadata_df['view'] == args.view)
        ].copy()

        val_df = combined_metadata_df[
            (combined_metadata_df['mri_id'].isin(val_mri_ids)) &
            (combined_metadata_df['view'] == args.view)
        ].copy()

        # Combine the file paths from both 'train' and 'val' subfolders
        def get_combined_path(row):
            if args.dataset_type == 'labrum':
                if os.path.exists(os.path.join(args.preprocessed_folder, 'train', row['mri_id'], f"{row['unique_id']}.npy")):
                    return os.path.join(args.preprocessed_folder, 'train', row['mri_id'], f"{row['unique_id']}.npy")
                else:
                    return os.path.join(args.preprocessed_folder, 'val', row['mri_id'], f"{row['unique_id']}.npy")
            elif args.dataset_type == 'MRNet':
                return os.path.join(args.preprocessed_folder, 'train', args.view, f"{row['mri_id']:04d}.npy")

        train_df['file_path'] = train_df.apply(get_combined_path, axis=1)
        val_df['file_path'] = val_df.apply(get_combined_path, axis=1)

        # Select the appropriate dataset class based on the model type and dataset type
        if args.dataset_type == 'labrum':
            if args.model_type == 'CNN3D':
                train_dataset = MRIDataset3D(train_df, args.preprocessed_folder, view=args.view, label_column=args.label_column, sequence_type=args.sequence_type, fat_sat=args.fat_sat, contrast_or_no=args.contrast_or_no, augment=args.augment, augment_factor=args.augment_factor, augment_factor_0=args.augment_factor_0, transform_val=args.transform_val, use_file_column=True)
                val_dataset = MRIDataset3D(val_df, args.preprocessed_folder, view=args.view, label_column=args.label_column, sequence_type=args.sequence_type, fat_sat=args.fat_sat, contrast_or_no=args.contrast_or_no, augment=False, transform_val=False, use_file_column=True)

                #for evaluation after the cycle finishes
                val_test_dataset = MRIDataset3D(val_df, args.preprocessed_folder, view=args.view, label_column=args.label_column, sequence_type=args.sequence_type, fat_sat=args.fat_sat, contrast_or_no=args.contrast_or_no, augment=False, transform_val=False, use_file_column=True, return_unique_id=True)

                holdout_test_dataset = MRIDataset3D(test_df, os.path.join(args.preprocessed_folder, 'test'), view=args.view, label_column=args.label_column, sequence_type=args.sequence_type, fat_sat=args.fat_sat, contrast_or_no=args.contrast_or_no, augment=False, transform_val=False, use_file_column=False, return_unique_id=True)

            else:
                train_dataset = MRIDataset2D(train_df, args.preprocessed_folder, view=args.view, label_column=args.label_column, sequence_type=args.sequence_type, fat_sat=args.fat_sat, contrast_or_no=args.contrast_or_no, augment=args.augment, augment_factor=args.augment_factor, augment_factor_0=args.augment_factor_0, transform_val=args.transform_val, use_file_column=True)
                val_dataset = MRIDataset2D(val_df, args.preprocessed_folder, view=args.view, label_column=args.label_column, sequence_type=args.sequence_type, fat_sat=args.fat_sat, contrast_or_no=args.contrast_or_no, augment=False, transform_val=False, use_file_column=True)
                
                #for evaluation after the cycle finishes
                val_test_dataset = MRIDataset2D(val_df, args.preprocessed_folder, view=args.view, label_column=args.label_column, sequence_type=args.sequence_type, fat_sat=args.fat_sat, contrast_or_no=args.contrast_or_no, augment=False, transform_val=False, use_file_column=True, return_unique_id=True)

                holdout_test_dataset = MRIDataset2D(test_df, os.path.join(args.preprocessed_folder, 'test'), view=args.view, label_column=args.label_column, sequence_type=args.sequence_type, fat_sat=args.fat_sat, contrast_or_no=args.contrast_or_no, augment=False, transform_val=False, use_file_column=False, return_unique_id=True)

        elif args.dataset_type == 'MRNet':
            if args.model_type == 'CNN3D':
                train_dataset = MRNetDataset3D(args.preprocessed_folder, train_df, view=args.view, label_column=args.label_column, augment=args.augment, augment_factor=args.augment_factor, transform_val=args.transform_val, use_file_column=True)
                val_dataset = MRNetDataset3D(args.preprocessed_folder, val_df, view=args.view, label_column=args.label_column, augment=False, transform_val=False, use_file_column=True)
                
                #for evaluation after the cycle finishes
                val_test_dataset = MRNetDataset3D(args.preprocessed_folder, val_df, view=args.view, label_column=args.label_column, augment=False, transform_val=False, use_file_column=True, return_unique_id=True)

                holdout_test_dataset = MRNetDataset3D(os.path.join(args.preprocessed_folder, 'test'), test_df, view=args.view, label_column=args.label_column, augment=False, transform_val=False, use_file_column=False, return_unique_id=True)

            else:
                train_dataset = MRNetDataset2D(args.preprocessed_folder, train_df, view=args.view, label_column=args.label_column, augment=args.augment, augment_factor=args.augment_factor, transform_val=args.transform_val, use_file_column=True)
                val_dataset = MRNetDataset2D(args.preprocessed_folder, val_df, view=args.view, label_column=args.label_column, augment=False, transform_val=False, use_file_column=True)
                
                #for evaluation after the cycle finishes
                val_test_dataset = MRNetDataset2D(args.preprocessed_folder, val_df, view=args.view, label_column=args.label_column, augment=False, transform_val=False, use_file_column=True, return_unique_id=True)

                holdout_test_dataset = MRNetDataset2D(os.path.join(args.preprocessed_folder, 'test'), test_df, view=args.view, label_column=args.label_column, augment=False, transform_val=False, use_file_column=False, return_unique_id=True)

        # Create DataLoaders for this cycle
        #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
        #val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
        #val_test_loader = DataLoader(val_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g
        )

        val_test_loader = DataLoader(
            val_test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g
        )

        holdout_test_loader = DataLoader(
            holdout_test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g
        )
        #test_dataset_size = len(holdout_test_loader.dataset)
        #print(f"Test dataset size: {test_dataset_size}")

        # Initialize TensorBoard writer
        log_dir = f"/gpfs/data/orthopedic-lab/tb_logs/{args.job_name}/cycle_{cycle+1}"
        writer = SummaryWriter(log_dir=log_dir)

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

        # Run the training function for this cycle
        cycle_save_path = os.path.join(save_path, f'cycle_{cycle + 1}')
        os.makedirs(cycle_save_path, exist_ok=True)

        model, val_acc = train_model(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer, 
            scheduler, 
            num_epochs=args.num_epochs, 
            device=device, 
            save_path=cycle_save_path, 
            pos_weight=pos_weight,
            start_epoch=start_epoch,
            T_max=T_max,
            save_checkpoints=save_checkpoints, 
            csv_path=csv_path,
            args=args, 
            return_acc=True, 
            writer=writer,
            early_stopping_patience=early_stopping_patience
        )

        # Evaluate the model on the validation set (use this for selecting best cycles)
        prob_csv_path = f'/gpfs/data/orthopedic-lab/ortho_ml/experiments/{args.job_name}_cycle_{cycle + 1}_val_probs.csv'
        y_true, y_pred, y_prob = evaluate_model(model, val_test_loader, criterion, device=device, csv_path=prob_csv_path, writer=writer)

        # Classification report
        print(classification_report(y_true, y_pred))

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        print("Val Confusion Matrix:\n", conf_matrix)

        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

        print("Model Training Complete")
        print(f'Val Accuracy: {accuracy:.4f}')
        print(f'Val Sensitivity (Recall): {recall:.4f}')
        print(f'Val Specificity: {specificity:.4f}')
        print(f'Val Precision: {precision:.4f}')
        print(f'Val F1 Score: {f1:.4f}')

        # Calculate AUC
        auc = roc_auc_score(y_true, y_prob)
        print(f"Final Val AUC: {auc:.4f}")

        # Log the final metrics to TensorBoard
        writer.add_scalar('Final Val Acc', accuracy)
        writer.add_scalar('Final Val Recall', recall)
        writer.add_scalar('Final Val Specificity', specificity)
        writer.add_scalar('Final Val Precision', precision)
        writer.add_scalar('Final Val F1_Score', f1)
        writer.add_scalar('Final Val AUC', auc)

        # Evaluate the model on the test set (so that probs csv is already saved when you select a cycle, but do not use test performance to select the cycle)
        test_prob_csv_path = f'/gpfs/data/orthopedic-lab/ortho_ml/experiments/{args.job_name}_cycle_{cycle + 1}_test_probs.csv'
        test_y_true, test_y_pred, test_y_prob = evaluate_model(model, holdout_test_loader, criterion, device=device, csv_path=test_prob_csv_path, writer=writer)

        # Classification report
        print(classification_report(test_y_true, test_y_pred))

        # Confusion matrix
        test_conf_matrix = confusion_matrix(test_y_true, test_y_pred)
        print("Test Confusion Matrix:\n", test_conf_matrix)

        # Metrics
        test_accuracy = accuracy_score(test_y_true, test_y_pred)
        test_recall = recall_score(test_y_true, test_y_pred)
        test_precision = precision_score(test_y_true, test_y_pred)
        test_f1 = f1_score(test_y_true, test_y_pred)
        test_specificity = test_conf_matrix[0, 0] / (test_conf_matrix[0, 0] + test_conf_matrix[0, 1])

        print("Model Training Complete")
        print(f'Test Accuracy: {test_accuracy:.4f}')
        print(f'Test Sensitivity (Recall): {test_recall:.4f}')
        print(f'Test Specificity: {test_specificity:.4f}')
        print(f'Test Precision: {test_precision:.4f}')
        print(f'Test F1 Score: {test_f1:.4f}')

        # Calculate AUC
        test_auc = roc_auc_score(test_y_true, test_y_prob)
        print(f"Test AUC: {test_auc:.4f}")

        # Log the final metrics to TensorBoard
        writer.add_scalar('Test Acc', test_accuracy)
        writer.add_scalar('Test Recall', test_recall)
        writer.add_scalar('Test Specificity', test_specificity)
        writer.add_scalar('Test Precision', test_precision)
        writer.add_scalar('Test F1_Score', test_f1)
        writer.add_scalar('Test AUC', test_auc)

        # Close the TensorBoard writer
        writer.close()

        # Store the validation accuracy for this cycle
        cycle_results.append(val_acc)
        print(f"Cycle {cycle + 1} Validation Accuracy: {val_acc:.4f}")
        
        # Save the final model
        torch.save(model.state_dict(), os.path.join(cycle_save_path, f'{args.job_name}_model_cycle_{cycle + 1}.pth'))

    return cycle_results

