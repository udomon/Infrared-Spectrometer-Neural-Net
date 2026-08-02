import torch
from sklearn.metrics import classification_report, f1_score


def train_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
        val_accuracy = evaluate_accuracy(model, val_loader, threshold=0.5)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2%}")

    return model


def evaluate_accuracy(model, loader, threshold=0.5):
    """Quick accuracy check — used for per-epoch monitoring."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for spectra, labels in loader:
            outputs = model(spectra).squeeze()
            predictions = (torch.sigmoid(outputs) > threshold).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def find_best_threshold(model, val_loader, thresholds=(0.30, 0.32, 0.35, 0.37, 0.40, 0.45, 0.50)):
    """Sweep thresholds on the validation set, return the one with best F1."""
    model.eval()
    val_outputs, val_labels = [], []

    with torch.no_grad():
        for spectra, labels in val_loader:
            outputs = torch.sigmoid(model(spectra).squeeze())
            val_outputs.extend(outputs.tolist())
            val_labels.extend(labels.tolist())

    best_threshold, best_f1 = 0.5, 0
    for threshold in thresholds:
        preds = [1 if p > threshold else 0 for p in val_outputs]
        f1 = f1_score(val_labels, preds)
        print(f"Threshold {threshold}: F1 = {f1:.3f}")
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    print(f"\nBest threshold: {best_threshold} (F1 = {best_f1:.3f})")
    return best_threshold


def evaluate_on_test(model, test_loader, threshold, target_names=('No carbonyl', 'Carbonyl')):
    """Final, one-time evaluation on the held-out test set."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for spectra, labels in test_loader:
            outputs = model(spectra).squeeze()
            predictions = (torch.sigmoid(outputs) > threshold).float()
            all_preds.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"Test accuracy: {accuracy:.2%}")
    print(classification_report(all_labels, all_preds, target_names=list(target_names)))

    return accuracy