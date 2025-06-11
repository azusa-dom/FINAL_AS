import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# We need to import your Dataset class to use it here
from .dataset import MRIDataset 


def get_kfold_strafied_sampler(data_dir, n_splits=5, batch_size=32):
    """
    A generator that finds pre-split fold data and yields DataLoaders for each fold.
    
    Args:
        data_dir (str): The directory where 'fold_X_train.csv' and 'fold_X_val.csv' are saved.
        n_splits (int): The number of folds to loop through.
        batch_size (int): The batch size for the DataLoader.
    
    Yields:
        (DataLoader, DataLoader): A tuple containing the train_loader and val_loader for a fold.
    """
    print(f"🔄 Loading {n_splits}-fold data from: {data_dir}")
    for i in range(n_splits):
        train_path = os.path.join(data_dir, f"fold_{i}_train.csv")
        val_path = os.path.join(data_dir, f"fold_{i}_val.csv")

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise FileNotFoundError(
                f"Data for fold {i} not found. Expected to find {train_path} and {val_path}"
            )

        # Create dataset objects using your MRIDataset class
        train_dataset = MRIDataset(csv_file=train_path)
        val_dataset = MRIDataset(csv_file=val_path)

        # Create DataLoader objects
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        yield train_loader, val_loader


def get_class_weights(dataset):
    """
    Computes class weights for handling imbalanced datasets.

    Args:
        dataset (Dataset): A PyTorch Dataset object which has a 'labels' attribute.
    
    Returns:
        torch.Tensor: A tensor containing the weight for each class.
    """
    # Access the labels from the dataset's underlying dataframe
    labels = dataset.df[dataset.label_column].to_numpy()
    unique_labels = np.unique(labels)
    
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    
    print(f"⚖️ Computed class weights: {class_weights}")
    return torch.tensor(class_weights, dtype=torch.float32)
