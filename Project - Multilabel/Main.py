import numpy as np
import pandas as pd
from rdkit import Chem
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch

from Data_Prep import load_metadata, build_dataset, augment_rare_labels, build_dataset_pretraining
from Dataset import IRSpectrumDataset
from Models import SpectrumCNN, SpectrumNet, SpectrumNetBigger
from Train import train_model, find_best_threshold, evaluate_on_test

META_PATH = r'10.22000-OGoEQGlsZGElrgst\10.22000-OGoEQGlsZGElrgst\data\dataset\meta_data.json'
EXP_FOLDER = r'10.22000-OGoEQGlsZGElrgst\10.22000-OGoEQGlsZGElrgst\data\dataset\JCAMP-DX Files\IR_data\exp'
IBMFILES = [
    r"C:\Users\mmudo\Documents\Programming\ML Datasets\IR_data_chunk001_of_009.parquet",
    r"C:\Users\mmudo\Documents\Programming\ML Datasets\IR_data_chunk002_of_009.parquet"
    ]


FUNCTIONAL_GROUPS = {
    'carbonyl': '[CX3]=[OX1]',
    'hydroxyl': '[OX2H]',
    'amine':    '[NX3;H2,H1;!$(NC=O)]',
    'nitrile':  '[CX2]#[NX1]',
    'aromatic': 'c1ccccc1',
    'alcohol': '[OX2H;!$(OC=O)]',            # alcohol O-H (excludes carboxylic acids)
    'carboxylic acid':    '[OX2H][CX3]=[OX1]',      
    'alkene':         '[CX3]=[CX3]',
    'C-O':         '[CX4][OX2]',                  # single C-O bond (ethers, alcohols)
}


def pretrain_on_ibm(compiled_patterns, target_grid):
    df = pd.concat([pd.read_parquet(f) for f in IBMFILES], ignore_index=True)
    X_pre, y_pre, skipped_pre = build_dataset_pretraining(df, target_grid, compiled_patterns)
    print(f"IBM dataset shape: {X_pre.shape}, Skipped: {skipped_pre}")

    X_pre_train, X_pre_val, y_pre_train, y_pre_val = train_test_split(X_pre, y_pre, test_size=0.1, random_state=0)

    train_loader = DataLoader(IRSpectrumDataset(X_pre_train, y_pre_train), batch_size = 64, shuffle=True)
    val_loader = DataLoader(IRSpectrumDataset(X_pre_val, y_pre_val), batch_size = 64, shuffle=False)

    pretrain_model = SpectrumCNN(num_labels=len(compiled_patterns))
    pretrain_model = train_model(pretrain_model, train_loader, val_loader, epochs=30, patience=8)

    torch.save(pretrain_model.state_dict(), 'pretrained_weights.pt')
    print("Pretrained weights saved.")

    #-------------------Training---------------------------------

def finetune_on_chemotion(compiled_patterns, target_grid):
    meta = load_metadata(META_PATH)
    X, y, skipped = build_dataset(meta, EXP_FOLDER, target_grid, pattern_dict=compiled_patterns)
    print(f"Chemotion dataset shape: {X.shape}, Skipped: {skipped}, Positives: {y.sum()}/{len(y)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=0)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    """augmentation only worsened training
    label_names = list(FUNCTIONAL_GROUPS.keys())
    rare_labels = ['carboxylic acid', 'amine', 'alcohol']

    X_train, y_train = augment_rare_labels(X_train, y_train, label_names, target_labels = rare_labels, num_augments = 4)

    print(f"Training set size after augmentation: {len(X_train)}")

    """

    """Pos weight only worsed training
    num_positive = y_train.sum(axis=0)          # shape: (num_labels,) — per-label positive counts
    num_negative = len(y_train) - num_positive
    pos_weight = torch.tensor(num_negative / num_positive, dtype=torch.float32)

    print("pos_weight per label:", dict(zip(FUNCTIONAL_GROUPS.keys(), pos_weight.tolist())))"""

    train_loader = DataLoader(IRSpectrumDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(IRSpectrumDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = DataLoader(IRSpectrumDataset(X_test, y_test), batch_size=32, shuffle=False)

    model = SpectrumCNN(num_labels=len(compiled_patterns))
    model.load_state_dict(torch.load('pretrained_weights.pt'))
    model = train_model(model, train_loader, val_loader, epochs=100, lr=0.0003, patience=15, use_focal_loss=False)

    label_names = list(compiled_patterns.keys())
    thresholds = find_best_threshold(model, val_loader, label_names=list(label_names))
    accuracies = evaluate_on_test(model, test_loader, thresholds, label_names=list(label_names))

    avg_accuracy = sum(accuracies.values()) / len(accuracies)
    print(f"Average test accuracy of model: {avg_accuracy:.2%}")

if __name__ == "__main__":
    compiled_patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in FUNCTIONAL_GROUPS.items()}
    target_grid = np.linspace(400, 4000, 1000)

    #pretrain_on_ibm(compiled_patterns, target_grid)
    finetune_on_chemotion(compiled_patterns, target_grid)