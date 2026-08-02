#from numba import jit,cuda

import torch
from sklearn.metrics import classification_report, f1_score
import copy

from Models import FocalLoss

#@jit(target_backend = 'cuda')
def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, patience = 15, use_focal_loss = True, gamma=2.0):
    if use_focal_loss:
        criterion = FocalLoss(gamma = gamma)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_accuracy = 0
    best_model_state = None
    epochs_without_improvement = 0


    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for spectra, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(spectra)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_accuracy = evaluate_accuracy(model, val_loader, threshold=0.5)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2%}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nNo improvement for {patience} epochs - stopping early at {epoch + 1}.")
            break

    print(f"\nBest validation accuracy: {best_val_accuracy:.2%}")
    model.load_state_dict(best_model_state)

    return model


def evaluate_accuracy(model, loader, threshold=0.5):
    """Quick accuracy check — used for per-epoch monitoring."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for spectra, labels in loader:
            outputs = model(spectra)
            predictions = (torch.sigmoid(outputs) > threshold).float()
            correct += (predictions == labels).sum().item()
            total += labels.numel() #counting each prediction label, not just molecule
    return correct / total


def find_best_threshold(model, val_loader, label_names, thresholds=(0.25, 0.30, 0.32, 0.35, 0.37, 0.40, 0.45, 0.50, 0.55)):
    """Sweep thresholds on the validation set, return the one with best F1."""
    model.eval()
    all_outputs, all_labels = [], []
    

    with torch.no_grad():
        for spectra, labels in val_loader:
            outputs = torch.sigmoid(model(spectra))
            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).numpy()  # shape: (num_samples, num_labels)
    all_labels = torch.cat(all_labels).numpy()

    best_thresholds = {}
    for i, name in enumerate(label_names):
        print(f"\n---Threshold calculation for {name}---")

        label_outputs = all_outputs[:, i] #one column at a time
        label_true = all_labels[:, i]

        best_threshold, best_f1 = 0.5, 0
        for threshold in thresholds:
            preds = (label_outputs > threshold).astype(int)
            f1 = f1_score(label_true, preds)
            print(f"Threshold {threshold}: F1 = {f1:.3f}")
            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold

        print(f"{name}: best threshold = {best_threshold} (F1 = {best_f1:.3f})")
        best_thresholds[name] = best_threshold

    return best_thresholds


def evaluate_on_test(model, test_loader, thresholds, label_names):
    """Final, one-time evaluation on the held-out test set."""
    model.eval()
    all_outputs, all_labels = [], []

    with torch.no_grad():
        for spectra, labels in test_loader:
            outputs = torch.sigmoid(model(spectra))
            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracies = {}
    for i, name in enumerate(label_names):
        threshold = thresholds[name]
        label_preds = (all_outputs[:, i] > threshold).astype(int)
        label_true = all_labels[:, i]

        accuracy = (label_preds == label_true).mean()

        print(f"\n--- {name} ---")
        print(f"Test accuracy: {accuracy:.2%}")
        print(classification_report(all_labels[:, i], label_preds, target_names=[f'No {name}', name]))

        accuracies[name] = accuracy

    return accuracies