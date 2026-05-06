"""
Search for more tasks where VPT might win.

VPT wins when: high attention variance (>0.3) AND low feature gap (<0.10)
= DINOv2 features are good, but attention routing needs task-specific changes

Strategy: test tasks where spatial layout differs strongly between classes
but DINOv2's feature space transfers well.
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import json
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from PIL import Image

from config import ExperimentConfig, setup_device, ensure_dirs
from exp5_task_structure import (extract_features_and_attention,
                                  measure_linear_probe_accuracy,
                                  measure_knn_accuracy,
                                  measure_attention_entropy,
                                  measure_attention_class_variance,
                                  measure_gradient_rank)


def get_extra_tasks(config):
    """Tasks that might have high attention variance + low feature gap."""

    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    gray_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tasks = {}

    # 1. Caltech101 — diverse objects, DINOv2 should be good
    #    Different objects at different positions/scales → maybe high attn_var
    try:
        from datasets import load_dataset
        ds = load_dataset('caltech101', split='train')
        class Caltech101DS(torch.utils.data.Dataset):
            def __init__(self, hf_ds, tfm, n):
                self.ds = hf_ds
                self.tfm = tfm
                self.n = min(n, len(hf_ds))
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                item = self.ds[idx]
                img = item['image']
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(np.array(img))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return self.tfm(img), item['label']
        tasks['caltech101'] = (Caltech101DS(ds, transform, 3000), 102)
        print(f"  caltech101: {len(ds)} samples")
    except Exception as e:
        print(f"  caltech101 failed: {e}")

    # 2. Sign Language MNIST — hand gestures with different spatial poses
    #    Download: kaggle datasets download -d datamunge/sign-language-mnist
    #    Or place sign_mnist_train.csv in ./data/sign_language_mnist/
    try:
        csv_path = None
        for p in ['./data/sign_language_mnist/sign_mnist_train.csv',
                  './data/sign_mnist_train.csv',
                  './sign_mnist_train.csv']:
            if os.path.exists(p):
                csv_path = p
                break

        if csv_path:
            import pandas as pd
            df = pd.read_csv(csv_path)
            class SignMNIST(torch.utils.data.Dataset):
                def __init__(self, df, tfm, n):
                    self.labels = df['label'].values[:n]
                    self.pixels = df.drop('label', axis=1).values[:n]
                    self.tfm = tfm
                def __len__(self):
                    return len(self.labels)
                def __getitem__(self, idx):
                    img = self.pixels[idx].reshape(28, 28).astype(np.uint8)
                    img = Image.fromarray(img, mode='L').convert('RGB')
                    img = self.tfm(img)
                    return img, self.labels[idx]

            tasks['sign_language_mnist'] = (SignMNIST(df, transform, 3000), 24)
            print(f"  sign_language_mnist: {len(df)} samples (24 hand gesture classes)")
        else:
            print("  sign_language_mnist: CSV not found. Download from:")
            print("    kaggle datasets download -d datamunge/sign-language-mnist")
    except Exception as e:
        print(f"  sign_language_mnist failed: {e}")

    # 3. Chinese MNIST — Chinese numeral characters (15 classes)
    #    Different characters have very different spatial strokes
    #    Download: kaggle datasets download -d gpreda/chinese-mnist
    try:
        csv_path = None
        for p in ['./data/chinese_mnist/chinese_mnist.csv',
                  './data/chinese_mnist.csv']:
            if os.path.exists(p):
                csv_path = p
                break

        img_dir = None
        for p in ['./data/chinese_mnist/data/data',
                  './data/chinese_mnist/data',
                  './data/chinese_mnist']:
            if os.path.exists(p) and any(f.endswith('.jpg') or f.endswith('.png')
                                         for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))):
                img_dir = p
                break

        if csv_path and img_dir:
            import pandas as pd
            df = pd.read_csv(csv_path)
            class ChineseMNIST(torch.utils.data.Dataset):
                def __init__(self, df, img_dir, tfm, n):
                    self.df = df.head(n)
                    self.img_dir = img_dir
                    self.tfm = tfm
                def __len__(self):
                    return len(self.df)
                def __getitem__(self, idx):
                    row = self.df.iloc[idx]
                    fname = f"input_{int(row['suite_id'])}_{int(row['sample_id'])}_{int(row['code'])}.jpg"
                    img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
                    img = self.tfm(img)
                    return img, int(row['code']) - 1  # 1-15 → 0-14

            tasks['chinese_mnist'] = (ChineseMNIST(df, img_dir, transform, 3000), 15)
            print(f"  chinese_mnist: {len(df)} samples (15 Chinese numeral classes)")
        else:
            print("  chinese_mnist: data not found. Download from:")
            print("    kaggle datasets download -d gpreda/chinese-mnist")
    except Exception as e:
        print(f"  chinese_mnist failed: {e}")

    # 3. SVHN (extra/easy split) — already tested but more data might change attn_var
    #    Skip — already have SVHN results

    # 4. EMNIST digits — like MNIST but more samples, different handwriting styles
    try:
        ds = datasets.EMNIST(root='./data', split='digits', train=True,
                             download=True, transform=gray_transform)
        indices = torch.randperm(len(ds))[:3000].tolist()
        tasks['emnist_digits'] = (Subset(ds, indices), 10)
        print(f"  emnist_digits: {len(ds)} samples")
    except Exception as e:
        print(f"  emnist_digits failed: {e}")

    # 5. EMNIST balanced — 47 classes (digits + upper + lower)
    try:
        ds = datasets.EMNIST(root='./data', split='balanced', train=True,
                             download=True, transform=gray_transform)
        indices = torch.randperm(len(ds))[:3000].tolist()
        tasks['emnist_balanced'] = (Subset(ds, indices), 47)
        print(f"  emnist_balanced: {len(ds)} samples")
    except Exception as e:
        print(f"  emnist_balanced failed: {e}")

    # 6. SUN397 — scene recognition, 397 categories
    #    Different scenes need attention to different regions
    try:
        ds = load_dataset('tanganke/sun397', split='train')
        class SUN397DS(torch.utils.data.Dataset):
            def __init__(self, hf_ds, tfm, n):
                self.ds = hf_ds
                self.tfm = tfm
                self.n = min(n, len(hf_ds))
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                item = self.ds[idx]
                img = item['image']
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(np.array(img))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return self.tfm(img), item['label']
        tasks['sun397'] = (SUN397DS(ds, transform, 3000), 397)
        print(f"  sun397: {len(ds)} samples")
    except Exception as e:
        print(f"  sun397 failed: {e}")

    # 7. Patch Camelyon variant — already low attn_var, skip

    # 8. RESISC45 — remote sensing, 45 scene classes
    try:
        ds = load_dataset('timm/resisc45', split='train')
        class RESISC45DS(torch.utils.data.Dataset):
            def __init__(self, hf_ds, tfm, n):
                self.ds = hf_ds
                self.tfm = tfm
                self.n = min(n, len(hf_ds))
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                item = self.ds[idx]
                img = item['image']
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(np.array(img))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return self.tfm(img), item['label']
        tasks['resisc45'] = (RESISC45DS(ds, transform, 3000), 45)
        print(f"  resisc45: {len(ds)} samples")
    except Exception as e:
        print(f"  resisc45 failed: {e}")

    return tasks


def main():
    config = ExperimentConfig()
    device = setup_device()

    print("Loading pretrained model...")
    model = timm.create_model(config.model_name, pretrained=True,
                               img_size=config.img_size).to(device)

    print("\nLoading extra tasks...")
    tasks = get_extra_tasks(config)

    all_results = {}

    for task_name, (dataset, n_classes) in tasks.items():
        print(f"\n{'='*55}")
        print(f"Task: {task_name} ({n_classes} classes)")
        print(f"{'='*55}")

        try:
            loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

            print("  Extracting features and attention...")
            features, labels, attention = extract_features_and_attention(
                model, loader, device, max_attn_batches=40)
            print(f"  {features.shape[0]} features, {features.shape[1]}d")

            lp_acc = measure_linear_probe_accuracy(features, labels, n_classes)
            knn_acc = measure_knn_accuracy(features, labels, k=5)
            mean_ent, max_ent, norm_ent = measure_attention_entropy(
                attention, config.num_layers)

            n_attn = attention[0].shape[0] if 0 in attention else 0
            attn_labels = labels[:n_attn]
            bv, wv, attn_ratio = measure_attention_class_variance(
                attention, attn_labels, config.num_layers)

            eff_rank, top4_frac = measure_gradient_rank(
                model, loader, device, n_classes, max_batches=3)

            gap = 1.0 - lp_acc

            # VPT-favorable?
            vpt_zone = "★ VPT ZONE" if attn_ratio > 0.3 and gap < 0.10 else ""

            print(f"  LP accuracy:     {lp_acc:.4f}")
            print(f"  Feature gap:     {gap:.4f}")
            print(f"  Attention var:   {attn_ratio:.4f}")
            print(f"  Gradient rank:   {eff_rank:.1f}")
            if vpt_zone:
                print(f"  >>> {vpt_zone} <<<")

            all_results[task_name] = {
                'n_classes': n_classes,
                'linear_probe_accuracy': lp_acc,
                'feature_gap': gap,
                'attention_class_variance_ratio': attn_ratio,
                'attention_entropy_normalized': norm_ent,
                'gradient_effective_rank': eff_rank,
                'gradient_top4_fraction': top4_frac,
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("EXTRA TASK STRUCTURE SUMMARY")
    print("=" * 70)
    print(f"\n  {'Task':<20s} {'Classes':>7s} {'LP Acc':>7s} {'Gap':>6s} "
          f"{'Attn Var':>8s} {'VPT zone?'}")
    print(f"  {'-'*65}")

    for name, r in sorted(all_results.items(), key=lambda x: -x[1]['attention_class_variance_ratio']):
        vpt = "★ YES" if r['attention_class_variance_ratio'] > 0.3 and r['feature_gap'] < 0.10 else ""
        print(f"  {name:<20s} {r['n_classes']:>7d} {r['linear_probe_accuracy']:>7.3f} "
              f"{r['feature_gap']:>6.3f} {r['attention_class_variance_ratio']:>8.4f}  {vpt}")

    print(f"\n  Reference (known VPT winner):")
    print(f"  {'mnist':<20s} {'10':>7s} {'0.945':>7s} {'0.055':>6s} {'0.4428':>8s}  ★ YES")

    # Merge with existing exp5
    exp5_path = os.path.join(config.output_dir, 'task_structure_analysis.json')
    if os.path.exists(exp5_path):
        with open(exp5_path) as f:
            existing = json.load(f)
        existing.update(all_results)
        with open(exp5_path, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"\n  Merged into {exp5_path}")


if __name__ == '__main__':
    main()
