import os
import json
import contextlib
import io
import numpy as np
import jcamp
from rdkit import Chem
import pandas as pd


def load_metadata(meta_path):
    with open(meta_path, 'r') as f:
        return json.load(f)


def find_dx_file(entry, exp_folder):
    """Given a metadata entry, return the full path to its .dx spectrum file, or None if not found."""
    datasets = entry.get('datasets', [])
    for dataset in datasets:
        for attachment in dataset.get('attacments', []):
            filename = attachment.get('filename', '')
            if filename.endswith('.dx') and not filename.endswith('.peak.jdx'):
                identifier = attachment['identifier']
                uuid = identifier.split('/')[-1]
                filepath = os.path.join(exp_folder, uuid)
                if os.path.isfile(filepath):
                    return filepath
    return None


def get_xy(spectrum):
    """Return x, y arrays from a jcamp dict, handling compound (LINK-type) files."""
    if 'children' in spectrum and len(spectrum['children']) > 0:
        block = spectrum['children'][0]
    else:
        block = spectrum
    return block['x'], block['y']


def normalise_spectrum(y):
    y_min, y_max = y.min(), y.max()
    if y_max - y_min == 0:
        return y
    return (y - y_min) / (y_max - y_min)

def augment_spectrum(y, noise_level=0.01, shift_max=3):
    """Create a slightly perturbed version of a spectrum — small noise + small shift."""
    y_aug = y.copy()
    y_aug += np.random.normal(0, noise_level, size=y.shape)
    shift = np.random.randint(-shift_max, shift_max + 1)
    y_aug = np.roll(y_aug, shift)
    return normalise_spectrum(y_aug)

def augment_rare_labels(X_train, y_train, label_names, target_labels, num_augments=4):
    """
    For each example that's positive for any label in target_labels,
    create `num_augments` augmented copies and add them to the training set.
    """
    X_aug_list = [X_train]
    y_aug_list = [y_train]

    target_indices = [label_names.index(name) for name in target_labels]

    for idx in range(len(X_train)):
        is_rare_positive = any(y_train[idx, i] == 1 for i in target_indices)
        if is_rare_positive:
            for _ in range(num_augments):
                X_aug_list.append(augment_spectrum(X_train[idx])[np.newaxis, :])
                y_aug_list.append(y_train[idx][np.newaxis, :])

    X_expanded = np.concatenate(X_aug_list, axis=0)
    y_expanded = np.concatenate(y_aug_list, axis=0)

    return X_expanded, y_expanded

def build_dataset(meta, exp_folder, target_grid, pattern_dict=None):
    """
    Loop through metadata, parse each spectrum, resample onto target_grid,
    and pair with its functional group label.
    Returns: X (spectra array), y (labels array), skipped (count of entries dropped)
    """
    X, y, skipped = [], [], 0

    for entry in meta:
        smiles = entry.get('cano_smiles')
        if not smiles:
            skipped += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped += 1
            continue

        path = find_dx_file(entry, exp_folder)
        if path is None:
            skipped += 1
            continue

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                spectrum = jcamp.readfile(path)
        except Exception:
            skipped += 1
            continue

        x_raw, y_raw = get_xy(spectrum)
        sort_idx = np.argsort(x_raw)
        y_resampled = np.interp(target_grid, x_raw[sort_idx], y_raw[sort_idx])
        y_resampled = normalise_spectrum(y_resampled)

        labels = [int(mol.HasSubstructMatch(pattern)) for pattern in pattern_dict.values()]

        X.append(y_resampled)
        y.append(labels)

    return np.array(X), np.array(y), skipped

def build_dataset_pretraining(df, target_grid, pattern_dict):
    X, y, skipped = [], [], 0

    for _, row in df.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped += 1
            continue

        x_raw = np.array(row['Frequency(cm^-1)'])
        y_raw = np.array(row['ir_spectra'])

        sort_idx = np.argsort(x_raw)
        y_resampled = np.interp(target_grid, x_raw[sort_idx], y_raw[sort_idx])
        y_resampled = normalise_spectrum(y_resampled)

        labels = [int(mol.HasSubstructMatch(pattern)) for pattern in pattern_dict.values()]

        X.append(y_resampled)
        y.append(labels)

    return np.array(X), np.array(y), skipped


def plot_spectrum(meta, exp_folder, index):
    import matplotlib.pyplot as plt

    entry = meta[index]
    path = find_dx_file(entry, exp_folder)
    if not path:
        print("No spectrum file found for this entry.")
        return

    spectrum = jcamp.readfile(path)
    x, y = get_xy(spectrum)

    plt.plot(x, y)
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Transmittance')
    plt.gca().invert_xaxis()
    plt.title('Test spectrum')
    plt.show()