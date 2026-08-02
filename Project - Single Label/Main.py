import numpy as np
from rdkit import Chem
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from Data_Prep import load_metadata, build_dataset
from Dataset import IRSpectrumDataset
from Models import SpectrumCNN
from Train import train_model, find_best_threshold, evaluate_on_test

META_PATH = r'10.22000-OGoEQGlsZGElrgst\10.22000-OGoEQGlsZGElrgst\data\dataset\meta_data.json'
EXP_FOLDER = r'10.22000-OGoEQGlsZGElrgst\10.22000-OGoEQGlsZGElrgst\data\dataset\JCAMP-DX Files\IR_data\exp'

FUNCTIONAL_GROUPS = {
    'carbonyl': '[CX3]=[OX1]',
    'hydroxyl': '[OX2H]',
    'amine':    '[NX3;H2,H1;!$(NC=O)]',
    'nitrile':  '[CX2]#[NX1]',
    'aromatic': 'c1ccccc1',
}


def main():
    meta = load_metadata(META_PATH)
    compiled_patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in FUNCTIONAL_GROUPS.items()}
    target_grid = np.linspace(400, 4000, 1000)

    X, y, skipped = build_dataset(meta, EXP_FOLDER, target_grid, label_name="carbonyl", pattern_dict=compiled_patterns)
    print(f"Dataset shape: {X.shape}, Skipped: {skipped}, Positives: {y.sum()}/{len(y)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=0, stratify=y_train)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    train_loader = DataLoader(IRSpectrumDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(IRSpectrumDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = DataLoader(IRSpectrumDataset(X_test, y_test), batch_size=32, shuffle=False)

    model = SpectrumCNN()
    model = train_model(model, train_loader, val_loader, epochs=20)

    threshold = find_best_threshold(model, val_loader)
    evaluate_on_test(model, test_loader, threshold)


if __name__ == "__main__":
    main()