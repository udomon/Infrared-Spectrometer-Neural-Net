import json
import numpy as np
from pathlib import Path
import pandas as pd
from rdkit import Chem
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
import jcamp

from Data_Prep import load_metadata, build_dataset, augment_rare_labels, build_dataset_pretraining
from Dataset import IRSpectrumDataset
from Models import SpectrumCNN, SpectrumNet, SpectrumNetBigger
from Train import train_model, find_best_threshold, evaluate_on_test
from Predict import predict_from_arrays, predict_from_file, print_predictions
from Digitise import digitise_spectrum

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
    'alcohol': '[OX2H;!$(OC=O)]',            
    'carboxylic acid':    '[OX2H][CX3]=[OX1]',      
    'alkene': '[CX3]=[CX3]',
    'C-O': '[CX4][OX2]',                 
}

# Keep this False for normal predictions.  Set it to True only when you want
# to retrain the network and overwrite trained_weights.pt / thresholds.json.
TRAIN_MODEL = False


def load_prediction_assets():
    """Load the trained model, saved validation thresholds, and output labels."""
    project_folder = Path(__file__).resolve().parent
    label_names = list(FUNCTIONAL_GROUPS.keys())
    threshold_path = next((path for path in (project_folder / 'thresholds.json', project_folder.parent / 'thresholds.json') if path.is_file()), None)
    weight_path = next((path for path in (project_folder / 'trained_weights.pt', project_folder.parent / 'trained_weights.pt') if path.is_file()), None)

    if threshold_path is None:
        raise FileNotFoundError(
            "thresholds.json is missing. Run training once with TRAIN_MODEL = True "
            "to save the per-label validation thresholds."
        )
    if weight_path is None:
        raise FileNotFoundError("trained_weights.pt is missing.")

    with threshold_path.open(encoding='utf-8') as threshold_file:
        saved_thresholds = json.load(threshold_file)
    thresholds = {label: float(saved_thresholds[label]) for label in label_names}

    model = SpectrumCNN(num_labels=len(label_names))
    model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
    model.eval()
    return model, thresholds, label_names


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

    """Pos weight only worsenedd training
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
    with (Path(__file__).resolve().parent / 'thresholds.json').open('w', encoding='utf-8') as threshold_file:
        json.dump({name: float(value) for name, value in thresholds.items()}, threshold_file, indent=2)
    print("Per-functional-group thresholds saved to thresholds.json")
    accuracies = evaluate_on_test(model, test_loader, thresholds, label_names=list(label_names))

    avg_accuracy = sum(accuracies.values()) / len(accuracies)
    print(f"Average test accuracy of model: {avg_accuracy:.2%}")

    torch.save(model.state_dict(), 'trained_weights.pt')
    print("Trained weights saved.")

    return thresholds, label_names

if __name__ == "__main__":
    compiled_patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in FUNCTIONAL_GROUPS.items()}
    target_grid = np.linspace(400, 4000, 1000)

    #pretrain_on_ibm(compiled_patterns, target_grid)
    if TRAIN_MODEL:
        thresholds, label_names = finetune_on_chemotion(compiled_patterns, target_grid)
        model = SpectrumCNN(num_labels=len(compiled_patterns))
        model.load_state_dict(torch.load('trained_weights.pt', map_location='cpu', weights_only=True))
        model.eval()
    else:
        model, thresholds, label_names = load_prediction_assets()


    # For JCAMP prediction, set JCAMP_PATH to a local spectrum file and uncomment:
    # result = predict_from_file(model, JCAMP_PATH, target_grid, label_names, thresholds, read_fn=jcamp.readfile)
    # print_predictions(result)

    #for image
    IMG_PATH = r"C:\Users\mmudo\OneDrive\Dokumenter\Supercurricular\Chemistry Neural Net\Project - Multilabel copy\Images\Amide.jpg"
    x_data, y_data = digitise_spectrum(IMG_PATH, x_values=(4000, 400), y_values=(0.0, 1.0))
    result = predict_from_arrays(model, x_data, y_data, target_grid, label_names, thresholds)
    print_predictions(result)

