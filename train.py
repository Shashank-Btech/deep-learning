"""
CIFAR-10 Image Classifier - Transfer Learning with ResNet-18
=============================================================
A complete deep learning pipeline demonstrating:
  - Transfer Learning (pretrained ImageNet -> CIFAR-10)
  - Two-phase fine-tuning (FC head -> backbone unfreeze)
  - Data Augmentation (flip, crop, color jitter, erasing)
  - Training curves, confusion matrix, classification report
  - Best-model checkpointing based on validation accuracy

Author : Shashank K
Framework: PyTorch
"""

import os
import sys
import json
import time
import datetime

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display required)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

# --- Constants ----------------------------------------------------------------
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
NUM_CLASSES = len(CLASSES)

# --- Helpers ------------------------------------------------------------------

class Logger:
    """Duplicate stdout to a log file so every print is saved."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        # Encode safely for Windows terminals (cp1252)
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            self.terminal.write(message.encode('ascii', errors='replace').decode('ascii'))
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def elapsed(start: float) -> str:
    """Return human-readable elapsed time."""
    m, s = divmod(int(time.time() - start), 60)
    return f"{m}m {s}s"


# --- Data Augmentation & Loaders ----------------------------------------------

def build_transforms():
    """
    Build training and test transforms for CIFAR-10 ResNet-20.
    Uses aggressive augmentation to improve generalization across all classes,
    especially dog, bird, and truck which historically show weaker scores.
    """
    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std  = [0.2023, 0.1994, 0.2010]

    train_tf = transforms.Compose([
        transforms.Resize((36, 36)),           # Slightly oversized for random crop variety
        transforms.RandomCrop(32, padding=4),  # Random crop restores 32x32
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15), # Helps dog/bird pose variation
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # Occlusion robustness
    ])

    test_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])

    return train_tf, test_tf


def load_data(batch_size: int, use_subset: bool = True):
    """
    Download CIFAR-10 and return DataLoaders.
    If use_subset=True, uses 10 000 train / 2 000 test for fast training.
    Pass --full flag to use the complete dataset.
    """
    train_tf, test_tf = build_transforms()

    train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    if use_subset:
        train_ds = Subset(train_ds, range(5_000))
        test_ds  = Subset(test_ds,  range(1_000))

    # num_workers=0 avoids spawning extra processes (reduces CPU/memory load on Windows)
    use_cuda = torch.cuda.is_available()
    num_workers = 2 if use_cuda else 0
    pin_mem = use_cuda

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_mem)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_mem)

    n_train = len(train_ds)
    n_test  = len(test_ds)
    return train_loader, test_loader, n_train, n_test


# --- Model --------------------------------------------------------------------

def build_model(device):
    """
    Transfer Learning Setup with 90%+ Accurate CIFAR-10 Model
    """
    import torch.hub
    print(">> Downloading Highly Accurate Base Model...")
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True, trust_repo=True)

    # Freeze earlier layers to maintain basic features, but unfreeze the majority of the network (layer2, layer3, fc)
    for name, param in model.named_parameters():
        if 'layer2' in name or 'layer3' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model.to(device)


def unfreeze_backbone(model):
    """
    Phase-2: Unfreeze the entire backbone for fine-tuning.
    """
    for param in model.parameters():
        param.requires_grad = True


# --- Training -----------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, n_samples):
    model.train()
    running_loss = 0.0
    running_correct = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()

    return running_loss / n_samples, running_correct / n_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_samples):
    model.eval()
    running_loss = 0.0
    running_correct = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()

    return running_loss / n_samples, running_correct / n_samples


def train(model, train_loader, test_loader, n_train, n_test, device, config):
    """
    Two-phase training loop:
      Phase 1 -- Train only the FC head (fast convergence)
      Phase 2 -- Unfreeze layer4 + FC, train with lower LR (fine-tuning)
    """
    # Class weights: boost dog (idx=5), bird (idx=2), truck (idx=9) for better per-class accuracy
    # CLASSES = [airplane(0), automobile(1), bird(2), cat(3), deer(4), dog(5), frog(6), horse(7), ship(8), truck(9)]
    class_weights = torch.tensor(
        [1.0, 1.0, 2.5, 1.2, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0],
        dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "phase": []}
    best_acc = 0.0
    epoch_global = 0

    phases = [
        {
            "name": "Phase 1 -- Majority Network Tuning (>90% Params)",
            "epochs": config["phase1_epochs"],
            "lr": config["phase1_lr"],
            "params": filter(lambda p: p.requires_grad, model.parameters()),
            "unfreeze": False,
        },
        {
            "name": "Phase 2 -- Fine-tune layer3 + layer4 + FC",
            "epochs": config["phase2_epochs"],
            "lr": config["phase2_lr"],
            "params": None,  # set after unfreeze
            "unfreeze": True,
        },
    ]

    for phase in phases:
        print(f"\n{'='*60}")
        print(f"  {phase['name']}")
        print(f"  LR = {phase['lr']}  |  Epochs = {phase['epochs']}")
        print(f"{'='*60}")

        if phase["unfreeze"]:
            unfreeze_backbone(model)
            # Collect all trainable parameters after unfreezing
            phase["params"] = filter(lambda p: p.requires_grad, model.parameters())

        optimizer = optim.Adam(phase["params"], lr=phase["lr"])
        scheduler = CosineAnnealingLR(optimizer, T_max=phase["epochs"], eta_min=1e-6)

        for ep in range(1, phase["epochs"] + 1):
            epoch_global += 1
            t0 = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, n_train
            )
            val_loss, val_acc = evaluate(
                model, test_loader, criterion, device, n_test
            )
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["phase"].append(1 if not phase["unfreeze"] else 2)

            marker = ""
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "outputs/best_model.pt")
                marker = " * saved"

            current_lr = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch_global:>2d} | "
                f"Train Loss {train_loss:.4f}  Acc {train_acc*100:5.1f}% | "
                f"Val Loss {val_loss:.4f}  Acc {val_acc*100:5.1f}% | "
                f"LR {current_lr:.6f} | {elapsed(t0)}{marker}"
            )

    print(f"\n[OK] Training complete -- Best Validation Accuracy: {best_acc*100:.2f}%")
    return history, best_acc


# --- Evaluation & Plotting ----------------------------------------------------

def plot_training_curves(history):
    """Save training/validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    phase_colors = ["#2196F3" if p == 1 else "#FF5722" for p in history["phase"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves - ResNet-18 Transfer Learning on CIFAR-10",
                 fontsize=14, fontweight="bold", y=1.02)

    # -- Loss --
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "o-", color="#1565C0", label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"],   "o-", color="#E53935", label="Val Loss",   linewidth=2)
    # Shade Phase 1 / Phase 2
    phase1_end = sum(1 for p in history["phase"] if p == 1)
    if phase1_end > 0 and phase1_end < len(epochs):
        ax.axvline(x=phase1_end + 0.5, color="gray", linestyle="--", alpha=0.6, label="Phase boundary")
        ax.axvspan(0.5, phase1_end + 0.5, alpha=0.06, color="blue")
        ax.axvspan(phase1_end + 0.5, len(epochs) + 0.5, alpha=0.06, color="orange")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Loss", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # -- Accuracy --
    ax = axes[1]
    ax.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", color="#1565C0",
            label="Train Acc", linewidth=2)
    ax.plot(epochs, [a * 100 for a in history["val_acc"]], "o-", color="#E53935",
            label="Val Acc", linewidth=2)
    if phase1_end > 0 and phase1_end < len(epochs):
        ax.axvline(x=phase1_end + 0.5, color="gray", linestyle="--", alpha=0.6, label="Phase boundary")
        ax.axvspan(0.5, phase1_end + 0.5, alpha=0.06, color="blue")
        ax.axvspan(phase1_end + 0.5, len(epochs) + 0.5, alpha=0.06, color="orange")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> Saved outputs/training_curves.png")


def plot_confusion_matrix(y_true, y_pred):
    """Save a confusion matrix heatmap with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    # Build annotation strings: count\n(pct%)
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.0f}%)"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Count"})
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Actual Label", fontsize=12)
    ax.set_title("Confusion Matrix - CIFAR-10 Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> Saved outputs/confusion_matrix.png")


def full_evaluation(model, test_loader, device):
    """Run full evaluation: classification report + confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n" + "="*42)
    print("       CLASSIFICATION REPORT")
    print("="*42)
    report = classification_report(all_labels, all_preds, target_names=CLASSES)
    print(report)

    plot_confusion_matrix(all_labels, all_preds)
    return all_labels, all_preds


# --- Sample Image Grid --------------------------------------------------------

def save_sample_grid(dataset, n=16):
    """Save a grid of sample CIFAR-10 images for the README."""
    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    fig.suptitle("Sample CIFAR-10 Images", fontsize=14, fontweight="bold")
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225],
    )
    for i, ax in enumerate(axes.flat):
        img, label = dataset[i]
        img = inv_normalize(img)
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(CLASSES[label], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/sample_images.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> Saved outputs/sample_images.png")


# --- Main ---------------------------------------------------------------------

def main():
    os.makedirs("outputs", exist_ok=True)
    logger = Logger("outputs/training_log.txt")
    sys.stdout = logger

    print("=" * 60)
    print("  CIFAR-10 Image Classifier - Transfer Learning Pipeline")
    print("=" * 60)
    print(f"  Date      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device    : {device}")

    # ── Configuration ──
    config = {
        "batch_size": 32,
        "phase1_epochs": 3,    # FC + majority backbone
        "phase1_lr": 2e-4,     # Slightly higher for faster initial convergence
        "phase2_epochs": 7,    # Full network fine-tune — more epochs = lower loss
        "phase2_lr": 1e-5,
    }

    use_full = "--full" in sys.argv
    use_subset = not use_full
    mode = "FULL (50K/10K)" if use_full else "SUBSET (10K/2K)"
    print(f"  Mode      : {mode}  (pass --full for complete dataset)")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Phase 1   : {config['phase1_epochs']} epochs, LR={config['phase1_lr']}")
    print(f"  Phase 2   : {config['phase2_epochs']} epochs, LR={config['phase2_lr']}")

    # ── Data ──
    print("\n>> Loading CIFAR-10 dataset...")
    t0 = time.time()
    train_loader, test_loader, n_train, n_test = load_data(
        config["batch_size"], use_subset=use_subset
    )
    print(f"  Train samples: {n_train:,}  |  Test samples: {n_test:,}  ({elapsed(t0)})")

    # Save a sample image grid
    raw_test = datasets.CIFAR10(root="./data", train=False, download=False,
                                transform=build_transforms()[1])
    save_sample_grid(raw_test)

    # ── Model ──
    print("\n>> Building ResNet-18 (Transfer Learning)...")
    model = build_model(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable (Ph1) : {trainable:,}  ({trainable/total_params*100:.1f}%)")

    # ── Train ──
    print("\n>> Starting training...")
    t0 = time.time()
    history, best_acc = train(
        model, train_loader, test_loader, n_train, n_test, device, config
    )
    print(f"\n  Total training time: {elapsed(t0)}")

    # ── Evaluate ──
    print("\n>> Running final evaluation on test set...")
    # Reload best weights for evaluation
    model.load_state_dict(torch.load("outputs/best_model.pt", map_location=device, weights_only=True))
    full_evaluation(model, test_loader, device)

    # ── Plots ──
    print("\n>> Generating plots...")
    plot_training_curves(history)

    # ── Save training history as JSON for the web UI chart ──
    history_json = {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "train_acc": [round(a * 100, 2) for a in history["train_acc"]],
        "val_acc": [round(a * 100, 2) for a in history["val_acc"]],
        "phase": history["phase"],
        "best_val_acc": round(best_acc * 100, 2),
    }
    with open("outputs/training_history.json", "w") as f:
        json.dump(history_json, f, indent=2)
    print("  -> Saved outputs/training_history.json")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  [OK] PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Best Val Accuracy : {best_acc*100:.2f}%")
    print(f"  Saved Model       : outputs/best_model.pt")
    print(f"  Training Curves   : outputs/training_curves.png")
    print(f"  Confusion Matrix  : outputs/confusion_matrix.png")
    print(f"  Sample Images     : outputs/sample_images.png")
    print(f"  Training Log      : outputs/training_log.txt")
    print("=" * 60)

    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()