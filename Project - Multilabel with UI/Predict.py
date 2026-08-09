# predict.py

import numpy as np
import torch
import contextlib
import io

from Data_Prep import get_xy, normalise_spectrum


def predict_from_arrays(model, x_data, y_data, target_grid, label_names, thresholds):
    """
    Run a trained model on raw (x, y) spectrum arrays and return predicted functional groups.

    x_data: array of wavenumbers
    y_data: array of intensity/transmittance values
    target_grid: the fixed wavenumber grid the model was trained on
    label_names: list of functional group names, in the same order the model outputs them
    thresholds: dict of {label_name: threshold}, from find_best_threshold
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    sort_idx = np.argsort(x_data)
    y_resampled = np.interp(target_grid, x_data[sort_idx], y_data[sort_idx])
    y_resampled = normalise_spectrum(y_resampled)

    x_tensor = torch.tensor(y_resampled, dtype=torch.float32).unsqueeze(0)  # add batch dimension

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_tensor)).squeeze(0)  # remove batch dimension

    results = {}
    for i, name in enumerate(label_names):
        prob = probs[i].item()
        threshold = thresholds[name]
        results[name] = {
            'present': prob > threshold,
            'probability': round(prob, 3),
        }

    return results


def predict_from_file(model, spectrum_path, target_grid, label_names, thresholds, read_fn):
    """
    Run a trained model on a spectrum file (e.g., JCAMP-DX) and return predicted functional groups.

    spectrum_path: path to the spectrum file
    read_fn: function to read the file into a dict with 'x'/'y' keys (e.g., jcamp.jcamp_reader)
    """
    with contextlib.redirect_stdout(io.StringIO()):
        spectrum = read_fn(spectrum_path)
    x_data, y_data = get_xy(spectrum)

    return predict_from_arrays(model, x_data, y_data, target_grid, label_names, thresholds)


def print_predictions(results):
    """Nicely print a predict_from_arrays/predict_from_file result dictionary."""
    for group, info in results.items():
        status = "YES" if info['present'] else "no"
        print(f"{group}: {status} (probability: {info['probability']})")