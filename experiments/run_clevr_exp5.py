"""Run exp5 task structure analysis on CLEVR only (from local Kaggle data)."""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import json
import os
import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from config import ExperimentConfig, setup_device, ensure_dirs
from exp5_task_structure import (extract_features_and_attention,
                                  measure_linear_probe_accuracy,
                                  measure_knn_accuracy,
                                  measure_attention_entropy,
                                  measure_attention_class_variance,
                                  measure_gradient_rank)


def load_clevr_local(config):
    """Load CLEVR from local Kaggle download with real object counts."""
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Find images
    search_paths = [
        './data/clevr/**/*.png',
        './data/CLEVR_v1.0/images/train/*.png',
        './data/clevr/CLEVR_v1.0/images/train/*.png',
        './data/clevr/images/train/*.png',
    ]
    all_images = []
    for pattern in search_paths:
        found = sorted(glob.glob(pattern, recursive=True))
        if found:
            all_images = found
            print(f"  Found {len(found)} CLEVR images at {pattern}")
            break

    if not all_images:
        print("  No CLEVR images found!")
        if os.path.exists('./data/clevr'):
            print("  Contents of ./data/clevr/:")
            for root, dirs, files in os.walk('./data/clevr'):
                level = root.replace('./data/clevr', '').count(os.sep)
                if level < 3:
                    indent = '  ' * level
                    print(f"    {indent}{os.path.basename(root)}/  ({len(files)} files)")
        return None

    # Find scene JSON for real labels
    scene_file = None
    for sp in ['./data/clevr/CLEVR_v1.0/scenes/CLEVR_train_scenes.json',
               './data/clevr/scenes/CLEVR_train_scenes.json',
               './data/CLEVR_v1.0/scenes/CLEVR_train_scenes.json']:
        if os.path.exists(sp):
            scene_file = sp
            break

    if scene_file:
        print(f"  Using real labels from {scene_file}")
        print(f"  Loading scene JSON (may take a moment)...")
        with open(scene_file) as f:
            scene_data = json.load(f)
        scenes = scene_data['scenes']
        del scene_data  # free ~1.5GB

        name_to_count = {}
        for s in scenes:
            count = min(len(s['objects']), 7)  # 0-7 = 8 classes
            name_to_count[s['image_filename']] = count
        del scenes  # free more memory

        # Filter images that have scene data
        valid_images = [p for p in all_images
                       if os.path.basename(p) in name_to_count]
        print(f"  {len(valid_images)} images with scene labels")

        # Print label distribution
        counts = {}
        for p in valid_images[:5000]:
            c = name_to_count[os.path.basename(p)]
            counts[c] = counts.get(c, 0) + 1
        print(f"  Label distribution (first 5000): {dict(sorted(counts.items()))}")
    else:
        print("  No scene JSON found — using pseudo-labels")
        valid_images = all_images
        name_to_count = {os.path.basename(p): hash(p) % 8 for p in all_images}

    # Use up to 2000 images (plenty for metrics, saves RAM)
    import random
    random.seed(42)
    random.shuffle(valid_images)
    valid_images = valid_images[:2000]

    class CLEVRDataset(torch.utils.data.Dataset):
        def __init__(self, paths, tfm, n2c):
            self.paths = paths
            self.transform = tfm
            self.n2c = n2c

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert('RGB')
            img = self.transform(img)
            label = self.n2c[os.path.basename(self.paths[idx])]
            return img, label

    return CLEVRDataset(valid_images, transform, name_to_count)


def main():
    config = ExperimentConfig()
    device = setup_device()

    print("Loading pretrained model...")
    model = timm.create_model(config.model_name, pretrained=True,
                               img_size=config.img_size).to(device)

    print("\nLoading CLEVR dataset...")
    dataset = load_clevr_local(config)
    if dataset is None:
        return

    print(f"  Total samples: {len(dataset)}")

    import gc
    gc.collect()  # free any leftover memory from data loading

    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # Extract features and attention (fewer attention batches to save RAM)
    print("\nExtracting features and attention maps...")
    features, labels, attention = extract_features_and_attention(
        model, loader, device, max_attn_batches=30)  # 30×64=1920 for attention
    print(f"  Features: {features.shape[0]} samples, {features.shape[1]}d")
    print(f"  Unique labels: {np.unique(labels)}")

    # All metrics
    print("\n--- CLEVR COUNT TASK STRUCTURE ---")

    lp_acc = measure_linear_probe_accuracy(features, labels, 8)
    print(f"  Linear probe accuracy:    {lp_acc:.4f}")

    knn_acc = measure_knn_accuracy(features, labels, k=5)
    print(f"  5-NN accuracy:            {knn_acc:.4f}")

    mean_ent, max_ent, norm_ent = measure_attention_entropy(
        attention, config.num_layers)
    print(f"  Attention entropy:        {mean_ent:.4f} / {max_ent:.4f} "
          f"(normalized: {norm_ent:.4f})")

    n_attn = attention[0].shape[0] if 0 in attention else 0
    attn_labels = labels[:n_attn]
    between_var, within_var, attn_ratio = measure_attention_class_variance(
        attention, attn_labels, config.num_layers)
    print(f"  Attention class variance: between={between_var:.6f}, "
          f"within={within_var:.6f}, ratio={attn_ratio:.4f}")

    print("\nComputing gradient rank...")
    eff_rank, top4_frac = measure_gradient_rank(
        model, loader, device, 8, max_batches=3)
    print(f"  Gradient effective rank:  {eff_rank:.1f}")
    print(f"  Gradient top-4 fraction:  {top4_frac:.4f}")

    feature_gap = 1.0 - lp_acc
    print(f"\n  Feature gap (1-LP_acc):   {feature_gap:.4f}")

    # Prediction
    if feature_gap < 0.10:
        pred = "LP"
    elif attn_ratio > 0.3 and feature_gap < 0.10:
        pred = "VPT"
    else:
        pred = "LoRA"
    print(f"  Predicted best method:    {pred}")

    # Save
    result = {
        'clevr_count': {
            'category': 'structured',
            'n_classes': 8,
            'linear_probe_accuracy': lp_acc,
            'knn_accuracy': knn_acc,
            'attention_entropy_normalized': norm_ent,
            'attention_class_variance_ratio': attn_ratio,
            'gradient_effective_rank': eff_rank,
            'gradient_top4_fraction': top4_frac,
            'feature_gap': feature_gap,
            'predicted_method': pred,
            'data_source': 'kaggle_local',
        }
    }

    # Merge with existing exp5 results
    exp5_path = os.path.join(config.output_dir, 'task_structure_analysis.json')
    if os.path.exists(exp5_path):
        with open(exp5_path) as f:
            existing = json.load(f)
        existing.update(result)
        result = existing

    with open(exp5_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to {exp5_path}")


if __name__ == '__main__':
    main()
