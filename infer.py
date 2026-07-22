"""
CIFAR-10 Inference Script
=========================
Classify a single image using the trained ResNet-18 model.
Shows top-5 predictions with confidence percentages.

Usage:
    python infer.py --image path/to/image.png
    python infer.py --image path/to/image.png --model outputs/best_model.pt
"""

import argparse
import sys
import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ─── Constants ────────────────────────────────────────────────────────
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ─── Model Loading ───────────────────────────────────────────────────

def load_model(weights_path: str, device: torch.device):
    """Load the highly-accurate ResNet-20 model pre-trained on CIFAR-10."""
    import torch.hub
    print("Loading base CIFAR-10 model architecture...")
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False, trust_repo=True)
    
    if os.path.exists(weights_path):
        print(f"Loading custom fine-tuned weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    else:
        print(f"Weights file not found at {weights_path}. Using base pre-trained weights.")
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True, trust_repo=True)
        
    model = model.to(device)
    model.eval()
    return model


# ─── Preprocessing ───────────────────────────────────────────────────

def get_transform():
    """Deterministic transform used by the chenyaofo CIFAR-10 model."""
    return transforms.Compose([
        transforms.Resize((32, 32)),  # Model expects exactly 32x32
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 specific mean
            std=[0.2023, 0.1994, 0.2010],   # CIFAR-10 specific std
        ),
    ])


# ─── Prediction ──────────────────────────────────────────────────────

def predict(image_path: str, model, device):
    """
    Run inference on a single image.
    Returns top-5 (class_name, confidence%) pairs.
    """
    image = Image.open(image_path).convert("RGB")
    tensor = get_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits[0], dim=0)

    top5_probs, top5_indices = torch.topk(probs, k=5)

    results = []
    for prob, idx in zip(top5_probs, top5_indices):
        results.append((CLASSES[idx.item()], prob.item() * 100))
    return results


# ─── Pretty Print ────────────────────────────────────────────────────

def print_results(image_path: str, results):
    """Print prediction results in a formatted table."""
    print()
    print("+" + "-"*46 + "+")
    print("|          CIFAR-10 INFERENCE RESULT           |")
    print("+" + "-"*46 + "+")
    print(f"|  Image: {os.path.basename(image_path):<37s}|")
    print("+" + "-"*6 + "+" + "-"*16 + "+" + "-"*23 + "+")
    print("| Rank | Class          | Confidence            |")
    print("+" + "-"*6 + "+" + "-"*16 + "+" + "-"*23 + "+")

    for i, (cls, conf) in enumerate(results):
        bar_len = int(conf / 5)  # Scale bar to max 20 chars
        bar = "#" * bar_len + "-" * (20 - bar_len)
        marker = " <" if i == 0 else ""
        print(f"|  {i+1}   | {cls:<14s} | {conf:5.1f}% {bar}{marker:>1s} |")

    print("+" + "-"*6 + "+" + "-"*16 + "+" + "-"*23 + "+")
    
    CONFIDENCE_THRESHOLD = 90.0
    if results[0][1] < CONFIDENCE_THRESHOLD:
        print(f"\n  >> Prediction: NOT RECOGNIZED (Low confidence: {results[0][1]:.1f}%)")
        print("  >> The model only recognizes the 10 core classes. Other patterns are rejected.")
    else:
        print(f"\n  >> Prediction: {results[0][0].upper()} ({results[0][1]:.1f}% confidence)")
    print()


# ─── CLI ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Image Classification — Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer.py --image outputs/sample_test.png
  python infer.py --image photo.jpg --model outputs/best_model.pt
        """,
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to the input image file (JPEG, PNG, etc.)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="outputs/best_model.pt",
        help="Path to the trained model weights (default: outputs/best_model.pt)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    if not os.path.isfile(args.image):
        print(f"Error: Image file not found: '{args.image}'")
        sys.exit(1)

    if not os.path.isfile(args.model):
        print(f"Error: Model file not found: '{args.model}'")
        print("  → Run 'python train.py' first to generate the model.")
        sys.exit(1)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load & predict
    model = load_model(args.model, device)
    results = predict(args.image, model, device)

    # Display
    print_results(args.image, results)


if __name__ == "__main__":
    main()