import os
import json
import jcamp
from rdkit import Chem
import numpy as np
import contextlib
import io
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import torch.nn as nn


#open the metadata
with open(r'10.22000-OGoEQGlsZGElrgst\10.22000-OGoEQGlsZGElrgst\data\dataset\meta_data.json', 'r') as f:
    meta = json.load(f)

exp_folder = r'10.22000-OGoEQGlsZGElrgst\10.22000-OGoEQGlsZGElrgst\data\dataset\JCAMP-DX Files\IR_data\exp'

def find_dx_file(entry, exp_folder):
    """Given a metadata entry, return the full path to its .dx spectrum file, or None if not found."""
    datasets = entry.get('datasets', [])
    
    for dataset in datasets:
        for attachment in dataset.get('attacments', []):
            filename = attachment.get('filename', '')
            if filename.endswith('.dx') and not filename.endswith('.peak.jdx'):
                identifier = attachment['identifier']
                uuid = identifier.split('/')[-1]  # take everything after the last '/'
                filepath = os.path.join(exp_folder, uuid)
                if os.path.isfile(filepath):
                    return filepath
    
    return None  # no matching .dx file found

def plot_spectrum(index):
    import matplotlib.pyplot as plt

    entry = meta[index]
    path = find_dx_file(entry, exp_folder)

    if path:
        spectrum = jcamp.readfile(path)
        print(spectrum.keys())
        print("x range:", spectrum["x"].min(), "to", spectrum["x"].max())
        print("y range:", spectrum["y"].min(), "to", spectrum["y"].max())
        print("Number of points:", len(spectrum['x']))

    plt.plot(spectrum['x'], spectrum['y'])
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Transmittance')
    plt.gca().invert_xaxis()  # IR spectra are conventionally plotted high-to-low wavenumber
    plt.title('Test spectrum')
    plt.show()

def normalise_spectrum(y):
    y_min, y_max = y.min(), y.max()
    if y_max - y_min == 0:
        return y
    return (y-y_min) / (y_max - y_min)

def get_xy(spectrum):
    """Return x, y arrays from a jcamp dict, handling compound (LINK-type) files."""
    if 'children' in spectrum and len(spectrum['children']) > 0:
        block = spectrum['children'][0]
    else:
        block = spectrum

    return block['x'], block['y']

def build_dataset(meta, exp_folder, target_grid, label_name="carbonyl", pattern_dict=None):
    #loop thru metadata and parse each spectrum and resample onto target_grid paired with functional group label
    #Returns: X (spectra array), y (labels array), skipped (num entries skipped)

    X = []
    y = []
    skipped = 0
    #skip_reasons = {'no_smiles': 0, 'bad_smiles': 0, 'no_file': 0, 'jcamp_error': 0}

    for entry in meta:
        smiles = entry.get('cano_smiles')
        if not smiles:
            skipped += 1
            #skip_reasons['no_smiles'] += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped += 1
            #skip_reasons['bad_smiles'] += 1
            continue

        path = find_dx_file(entry, exp_folder)
        if path is None:
            skipped += 1
            #skip_reasons['no_file'] += 1
            continue

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                spectrum = jcamp.readfile(path)
            
        except Exception as e:
            skipped += 1
            #skip_reasons['jcamp_error'] += 1
            #if skipped <= 5:
            #    print("Error:", type(e).__name__, e)
            continue

        x_raw, y_raw = get_xy(spectrum)

        sort_idx = np.argsort(x_raw)
        y_resampled = np.interp(target_grid, x_raw[sort_idx], y_raw[sort_idx])
        y_resampled = normalise_spectrum(y_resampled)

        label = int(mol.HasSubstructMatch(pattern_dict[label_name]))

        X.append(y_resampled)
        y.append(label)

    return np.array(X), np.array(y), skipped

class IRSpectrumDataset(Dataset): #compatible with DataLoader
    def __init__(self, X, y):
        #X: numpy array of shape (num_samples, nump_points) - resampled spectra
        #y: numpy array of shape (num_samples,) - labels [0,1]

        self.X = torch.tensor(X, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)

    def __len__(self): #get length
        return len(self.X)

    def __getitem__(self, idx): #get example
        return self.X[idx], self.y[idx]

class SpectrumNet(nn.Module):
    def __init__(self, input_size=1000):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)  # single output — binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # raw logit — no sigmoid here, BCEWithLogitsLoss applies it internally

target_grid = np.linspace(400,4000,1000)

functional_groups = {
    'carbonyl':    '[CX3]=[OX1]',
    'hydroxyl':    '[OX2H]',
    'amine':       '[NX3;H2,H1;!$(NC=O)]',
    'nitrile':     '[CX2]#[NX1]',
    'aromatic':    'c1ccccc1',
}

# Pre-compiled the SMARTS patterns once, rather than re-parsing them every loop iteration
compiled_patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in functional_groups.items()}

X, y, skipped = build_dataset(meta, exp_folder, target_grid, label_name="carbonyl", pattern_dict = compiled_patterns)

"""print(f"Dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Skipped: {skipped}")
print(f"Positive (carbonyl) examples: {y.sum()} / {len(y)}")"""

#train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Then split into train/validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=0, stratify=y_train)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

train_dataset = IRSpectrumDataset(X_train, y_train)
val_dataset = IRSpectrumDataset(X_val, y_val)
test_dataset = IRSpectrumDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = SpectrumNet()

criterion = nn.BCEWithLogitsLoss() #binary cross entropy -> yes no outcome
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#training
epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for spectra, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(spectra).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    #validation check#
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for spectra, labels in val_loader:
            outputs = model(spectra).squeeze()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2%}")

#find the right threshold
def find_threshold():
    model.eval()
    val_outputs = []
    val_labels_list = []

    with torch.no_grad():
        for spectra, labels in val_loader:
            outputs = torch.sigmoid(model(spectra).squeeze())
            val_outputs.extend(outputs.tolist())
            val_labels_list.extend(labels.tolist())

    best_threshold = 0.5
    best_f1 = 0
    for threshold in [0.30, 0.32, 0.35, 0.37, 0.40, 0.45, 0.50]:
        preds = [1 if p > threshold else 0 for p in val_outputs]
        f1 = f1_score(val_labels_list, preds)
        print(f"Threshold {threshold}: F1 = {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\nBest threshold: {best_threshold} (F1 = {best_f1:.3f})")
    return best_threshold

threshold = find_threshold()

#testing
model.eval()
correct = 0 #counting total correct tests
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for spectra, labels in test_loader:
        outputs = model(spectra).squeeze()
        predictions = (torch.sigmoid(outputs) > threshold).float()
        correct += (predictions == labels).sum().item() #sum of true 
        total += labels.size(0) 
        all_preds.extend(predictions.tolist())
        all_labels.extend(labels.tolist())

accuracy = correct / total
print(f"Test accuracy: {accuracy:.2%}")

print(classification_report(all_labels, all_preds, target_names=['No carbonyl', 'Carbonyl']))
