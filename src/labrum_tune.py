import argparse
from models import CNN3D, ResNet50, AlexNet, VisionTransformer, SwinTransformerV1, SwinTransformerV2, ResNet34, DenseNet, EfficientNet
from training_functions import train_model, evaluate_model, load_checkpoint, save_checkpoint, find_latest_checkpoint, compute_pos_weight, load_model_weights
from loader import prepare_and_create_loaders_from_params,  load_metadata
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
import optuna
from optuna.trial import TrialState
import pickle
from torch.utils.tensorboard import SummaryWriter
import joblib

SEED = 42  
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def save_study(study, filename):
    joblib.dump(study, filename)

def load_study(filename):
    return joblib.load(filename)

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

def objective(trial):
    # Define hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-8, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    scheduler_type = trial.suggest_categorical('scheduler', ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CyclicLR'])
    augment_factor = 10 
    augment_factor_0 = augment_factor 

    # Suggest categorical hyperparameters for sequence_type, fat_sat, and contrast_or_no
    sequence_type = args.sequence_type 
    fat_sat = args.fat_sat
    contrast_or_no = args.contrast_or_no 

    # Arguments from argparse or function call
    num_workers=4
    batch_size = args.batch_size 
    preprocessed_folder = args.preprocessed_folder
    label_column = args.label_column
    view = args.view
    num_epochs = args.num_epochs
    model_type = args.model_type
    job_name = args.job_name
    augment = args.augment
    transform_val = args.transform_val

    # Create a unique TensorBoard writer for each trial
    log_dir = f"/gpfs/data/orthopedic-lab/tb_logs/{job_name}/trial_{trial.number}" #update this with your 
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Starting trial {trial.number} with parameters: {trial.params}")

    # Set up checkpoint directory
    checkpoint_dir = os.path.join(preprocessed_folder, job_name, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

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

    # Prepare datasets using the unified loader
    combined_train_loader, combined_val_loader, combined_test_loader = prepare_and_create_loaders_from_params(
        preprocessed_folder=preprocessed_folder,
        label_column=label_column,
        view=view,
        batch_size=batch_size,
        augment=augment,
        augment_factor=augment_factor,
        augment_factor_0=augment_factor_0,
        transform_val=transform_val,
        model_type=model_type,
        dataset_type=args.dataset_type,
        sequence_type=sequence_type,
        fat_sat=fat_sat,
        contrast_or_no=contrast_or_no,
        num_workers=num_workers
    )

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
    if model_type == 'ResNet50':
        model = ResNet50(dropout_rate=dropout_rate).to(device)
    elif model_type == 'AlexNet':
        model = AlexNet(dropout_rate=dropout_rate).to(device)
    elif model_type == 'CNN3D':
        model = CNN3D(dropout_rate=dropout_rate).to(device)
    elif model_type == 'VisionTransformer':
        model = VisionTransformer(dropout_rate=dropout_rate).to(device)
    elif model_type == 'SwinTransformerV1':
        model = SwinTransformerV1(dropout_rate=dropout_rate).to(device)
    elif model_type == 'SwinTransformerV2':
        model = SwinTransformerV2(dropout_rate=dropout_rate).to(device)
    elif model_type == 'EfficientNet':
        model = EfficientNet(dropout_rate=dropout_rate).to(device)
    elif model_type == 'DenseNet':
        model = DenseNet(dropout_rate=dropout_rate).to(device)
    elif model_type == 'ResNet34':
        model = ResNet34(dropout_rate=dropout_rate).to(device)
    else:
        print("Unrecognized model type")

    # Print model type
    print("Model type:", f"{model_type}")

    # Calculate pos_weight based on input argument and dataset type
    if args.pos_weight.lower() == "automatic":
        if args.dataset_type == 'labrum':
            # For labrum dataset
            train_metadata_df = load_metadata('train', args.preprocessed_folder)
            pos_weight = compute_pos_weight(train_metadata_df, args.label_column, contrast_or_no=contrast_or_no, view=view).to(device)
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


    augment_status = augment
    if augment_status:
        print('Using data augmentation')
    else:
        print('Not using data augmentation')

    if transform_val:
        print('Using transforms')
    else:
        print('Not using transforms')

    # Initialize BCEWithLogitsLoss with pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Define optimizer with new learning rate and weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Add a learning rate scheduler
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        T_max = 0 #pass placeholder into train_model for hparam logging
    elif scheduler_type == 'CosineAnnealingLR':
        T_max = 10 #(num_epochs // 5))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == 'CyclicLR':
        base_lr = lr * 0.1  # Set base_lr as 10% of initial learning rate
        max_lr = lr * 10    # Set max_lr as 10x the initial learning rate
        step_size_up = 2000  # Number of iterations to reach max_lr
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, mode='triangular')
        T_max = 0  # Placeholder since CyclicLR doesn't use T_max
    else:
        raise ValueError(f"Unrecognized scheduler type: {scheduler_type}")

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
        start_epoch = 0
        best_acc = 0.0
        print("No checkpoint found, starting training from scratch.")


    # Define CSV path for tuning results based on dataset type
    if args.dataset_type == 'labrum':
        csv_path = "/gpfs/data/orthopedic-lab/ortho_ml/experiments/labrum_tune_results.csv"
    elif args.dataset_type == 'MRNet':
        csv_path = "/gpfs/data/orthopedic-lab/ortho_ml/experiments/MRNet_tune_results.csv"

    # Train the model with checkpoints disabled
    model, val_acc = train_model(
        model=model,
        train_loader=combined_train_loader,
        val_loader=combined_val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        pos_weight=pos_weight,
        device=device,
        save_path=checkpoint_dir,
        start_epoch=start_epoch,
        best_acc=best_acc,
        save_checkpoints=False,
        csv_path=csv_path,
        args=args,
        trial=trial,
        return_acc=True,
        writer=writer,
        early_stopping_patience=10  # Added early stopping parameter
    )

    prob_csv_path = f'/gpfs/data/orthopedic-lab/ortho_ml/experiments/{args.job_name}_trial_{trial.number}_probs.csv'

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

    # Only save the final model if it has a test AUC > 0.70
    if (auc > 0.70):
        torch.save(model.state_dict(), os.path.join(args.preprocessed_folder, args.job_name, f'{args.job_name}_trial_{trial.number}_model.pth'))
        print('Test AUC > 0.70, so model state saved.')

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

    # Close the TensorBoard writer
    writer.close()

    return val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune hyperparameters for 3D CNN on labrum dataset")
    parser.add_argument('--preprocessed_folder', type=str, required=True, help='Preprocessed folder for labrum dataset')
    parser.add_argument('--label_column', type=str, required=True, help='Label column for classification')
    parser.add_argument('--view', type=str, required=True, help='Selected view (axial, sagittal, coronal, ABERS, all)')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size; 1 for all 2D models, CNN3D must have > 1')
    parser.add_argument('--sequence_type', type=str, default='all', help='Sequence type to include (e.g., T1, T2, all)')
    parser.add_argument('--fat_sat', type=str, default='all', help='Fat saturation to include (Yes, No, all)')
    parser.add_argument('--contrast_or_no', type=str, default='all', help='Contrast type to include (WO, W, WWO, all)')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs for training')
    parser.add_argument('--model_type', type=str, required=True, help='Model type')
    parser.add_argument('--model_weights', type=str, default=None, help='Path to model weights to initialize from')
    parser.add_argument('--job_name', type=str, required=True, help='Job name for checkpoint directory')
    parser.add_argument('--augment', type=str2bool, nargs='?', const=True, default=False, help="Whether to apply data augmentation during training")
    parser.add_argument('--augment_factor', type=int, required=True)
    parser.add_argument('--augment_factor_0', type=int, default=1)
    parser.add_argument('--transform_val', type=str2bool, nargs='?', const=True, default=True, help="Whether to apply data augmentation during training")
    parser.add_argument('--ret_val_probs', type=str2bool, nargs='?', const=True, default=False, help="Whether to return inference probabilities on validation set")
    parser.add_argument('--pos_weight', type=str, required=True, help='Set pos_weight to "automatic" or an integer value.')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials for hyperparameter tuning')
    parser.add_argument('--dataset_type', type=str, required=True, choices=['labrum', 'MRNet'], help='Dataset type (labrum or MRNet)')

    args = parser.parse_args()

    # Define the path where you want to save the study
    study_save_path = f"/gpfs/data/orthopedic-lab/ortho_ml/experiments/{args.job_name}_optuna_study.pkl"

    # Callback function to save the study after each trial
    def save_study_callback(study, trial):
        save_study(study, study_save_path)
        print(f"Study saved to {study_save_path} after trial {trial.number}")

    # Check if the study save file exists
    if os.path.exists(study_save_path):
        print("Loading existing study...")
        study = load_study(study_save_path)
        #print(f"Number of completed trials in the loaded study: {len(study.trials)}")
        #print(f"Loaded study: {len(study.trials)} trials")

        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"Pruned trials: {len(pruned_trials)}, Completed trials: {len(completed_trials)}")

    else:
        print("Creating a new study...")
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=args.num_epochs, reduction_factor=3))
        #study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())

    # Print information about the last study, if it exists
    if len(study.trials) > 0:
        last_trial = study.trials[-1]
        print(f"Last completed trial number: {last_trial.number}")
        print(f"  Trial Value: {last_trial.value}")
        print(f"  Trial Parameters: {last_trial.params}")
    else:
        print("No trials have been completed yet.")

    # Optimize the study
    try:
        study.optimize(objective, n_trials=args.n_trials, callbacks=[save_study_callback])
    finally:
        # Save the study after each run or if interrupted
        save_study(study, study_save_path)
        print(f"Study saved to {study_save_path} after trial {len(study.trials)}")

    # Save the study trials to a CSV file for later visualization
    study_trials_csv_path = f"/gpfs/data/orthopedic-lab/ortho_ml/experiments/{args.job_name}_trials.csv"
    df = study.trials_dataframe()
    df.to_csv(study_trials_csv_path, index=False)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')


