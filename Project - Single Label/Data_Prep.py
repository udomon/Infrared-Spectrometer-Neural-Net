import os
import json
import contextlib
import io
import numpy as np
import jcamp
from rdkit import Chem


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


def build_dataset(meta, exp_folder, target_grid, label_name="carbonyl", pattern_dict=None):
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

        label = int(mol.HasSubstructMatch(pattern_dict[label_name]))

        X.append(y_resampled)
        y.append(label)

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