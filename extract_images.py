import os
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image

def main():
    print("Extracting CIFAR-10 images to disk...")
    # Load dataset without any tensor transformations so we get PIL images
    dataset = datasets.CIFAR10(root="./data", train=True, download=True)
    
    classes = dataset.classes
    
    # Create output directory
    out_dir = "data/extracted_images"
    os.makedirs(out_dir, exist_ok=True)
    
    # Create class subdirectories
    for c in classes:
        os.makedirs(os.path.join(out_dir, c), exist_ok=True)
    
    # Extract 50 images per class so it doesn't take forever (500 images total)
    counts = {c: 0 for c in classes}
    target_per_class = 50
    
    for i in range(len(dataset)):
        img, label_idx = dataset[i]
        class_name = classes[label_idx]
        
        if counts[class_name] < target_per_class:
            img_path = os.path.join(out_dir, class_name, f"{class_name}_{counts[class_name]}.png")
            img.save(img_path)
            counts[class_name] += 1
            
        # Check if we have enough of all classes
        if all(c == target_per_class for c in counts.values()):
            break

    print(f"Done! Extracted 50 images per class into {out_dir}")

if __name__ == "__main__":
    main()
