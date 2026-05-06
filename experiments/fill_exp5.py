"""Fill missing tasks in exp5 JSON. Uses 2000 samples max per task."""
import sys
sys.path.insert(0, '.')

import torch
import timm
import numpy as np
import json
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

from config import ExperimentConfig, setup_device
from exp5_task_structure import (extract_features_and_attention,
                                  measure_linear_probe_accuracy,
                                  measure_knn_accuracy,
                                  measure_attention_entropy,
                                  measure_attention_class_variance,
                                  measure_gradient_rank)

config = ExperimentConfig()
device = setup_device()

# Load existing
exp5_path = os.path.join(config.output_dir, 'task_structure_analysis.json')
existing = {}
if os.path.exists(exp5_path):
    with open(exp5_path) as f:
        existing = json.load(f)
print(f"Existing exp5 tasks: {sorted(existing.keys())}")

# All tasks we need
all_tasks = {
    'cifar10': (10, 'natural'),
    'cifar100': (100, 'natural'),
    'dtd': (47, 'natural'),
    'oxford_flowers102': (102, 'natural'),
    'oxford_iiit_pet': (37, 'natural'),
    'food101': (101, 'natural'),
    'stl10': (10, 'natural'),
    'fgvc_aircraft': (100, 'natural'),
    'eurosat': (10, 'specialized'),
    'pcam': (2, 'specialized'),
    'country211': (211, 'specialized'),
    'svhn': (10, 'structured'),
    'gtsrb': (43, 'structured'),
    'mnist': (10, 'structured'),
    'fashionmnist': (10, 'structured'),
    'emnist_letters': (26, 'structured'),
    'emnist_digits': (10, 'structured'),
    'rendered_sst2': (2, 'structured'),
    'clevr_count': (8, 'structured'),
}

missing = [t for t in all_tasks if t not in existing]
print(f"Missing: {missing}")

if not missing:
    print("All tasks present! Nothing to do.")
    exit()

# Load model
print("Loading model...")
model = timm.create_model(config.model_name, pretrained=True,
                           img_size=config.img_size).to(device)

gray_tfm = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
rgb_tfm = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

MAX_SAMPLES = 2000

def get_dataset(name):
    """Load dataset with max 2000 samples."""
    loaders = {
        'cifar10': lambda: datasets.CIFAR10('./data', True, download=True, transform=rgb_tfm),
        'cifar100': lambda: datasets.CIFAR100('./data', True, download=True, transform=rgb_tfm),
        'dtd': lambda: datasets.DTD('./data', 'train', download=True, transform=rgb_tfm),
        'oxford_flowers102': lambda: datasets.Flowers102('./data', 'train', download=True, transform=rgb_tfm),
        'oxford_iiit_pet': lambda: datasets.OxfordIIITPet('./data', 'trainval', download=True, transform=rgb_tfm),
        'food101': lambda: datasets.Food101('./data', 'train', download=True, transform=rgb_tfm),
        'stl10': lambda: datasets.STL10('./data', 'train', download=True, transform=rgb_tfm),
        'fgvc_aircraft': lambda: datasets.FGVCAircraft('./data', 'train', download=True, transform=rgb_tfm),
        'eurosat': lambda: datasets.EuroSAT('./data', download=True, transform=rgb_tfm),
        'pcam': lambda: datasets.PCAM('./data', 'train', download=True, transform=rgb_tfm),
        'country211': lambda: datasets.Country211('./data', 'train', download=True, transform=rgb_tfm),
        'svhn': lambda: datasets.SVHN('./data', 'train', download=True, transform=rgb_tfm),
        'gtsrb': lambda: datasets.GTSRB('./data', 'train', download=True, transform=rgb_tfm),
        'mnist': lambda: datasets.MNIST('./data', True, download=True, transform=gray_tfm),
        'fashionmnist': lambda: datasets.FashionMNIST('./data', True, download=True, transform=gray_tfm),
        'emnist_letters': lambda: datasets.EMNIST('./data', 'letters', True, download=True, transform=gray_tfm),
        'emnist_digits': lambda: datasets.EMNIST('./data', 'digits', True, download=True, transform=gray_tfm),
        'rendered_sst2': lambda: datasets.RenderedSST2('./data', 'train', download=True, transform=rgb_tfm),
    }
    if name not in loaders:
        return None
    ds = loaders[name]()
    if len(ds) > MAX_SAMPLES:
        indices = torch.randperm(len(ds))[:MAX_SAMPLES].tolist()
        ds = Subset(ds, indices)
    return ds


for task_name in missing:
    n_classes, category = all_tasks[task_name]
    print(f"\n{'='*50}")
    print(f"Task: {task_name} ({n_classes} classes)")
    print(f"{'='*50}")

    try:
        ds = get_dataset(task_name)
        if ds is None:
            print("  No loader available, skipping")
            continue

        print(f"  Loaded {len(ds)} samples")

        # Handle EMNIST label shift (1-indexed)
        needs_label_fix = task_name == 'emnist_letters'

        class LabelFixWrapper(torch.utils.data.Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                img, label = self.ds[idx]
                return img, label - 1

        if needs_label_fix:
            ds = LabelFixWrapper(ds)

        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        features, labels, attention = extract_features_and_attention(
            model, loader, device, max_attn_batches=30)
        print(f"  Extracted {features.shape[0]} features")

        lp_acc = measure_linear_probe_accuracy(features, labels, n_classes)
        knn_acc = measure_knn_accuracy(features, labels, k=5)
        mean_ent, max_ent, norm_ent = measure_attention_entropy(attention, config.num_layers)

        n_attn = attention[0].shape[0] if 0 in attention else 0
        attn_labels = labels[:n_attn]
        bv, wv, attn_ratio = measure_attention_class_variance(attention, attn_labels, config.num_layers)

        eff_rank, top4_frac = measure_gradient_rank(model, loader, device, n_classes, max_batches=2)

        gap = 1.0 - lp_acc

        print(f"  LP={lp_acc:.3f}  gap={gap:.3f}  attn_var={attn_ratio:.4f}  grad_rank={eff_rank:.1f}")

        existing[task_name] = {
            'category': category,
            'n_classes': n_classes,
            'linear_probe_accuracy': float(lp_acc),
            'knn_accuracy': float(knn_acc),
            'attention_entropy_normalized': float(norm_ent),
            'attention_class_variance_ratio': float(attn_ratio),
            'gradient_effective_rank': float(eff_rank),
            'gradient_top4_fraction': float(top4_frac),
            'feature_gap': float(gap),
        }

        # Save after each task
        with open(exp5_path, 'w') as f:
            json.dump(existing, f, indent=2)

        # Free attention memory
        del features, labels, attention
        import gc; gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        torch.cuda.empty_cache()

print(f"\nDone. {len(existing)} tasks in {exp5_path}")
