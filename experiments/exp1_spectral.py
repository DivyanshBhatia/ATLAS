"""
Experiment 1: Spectral Profile Analysis

Purpose: Validate the core assumption of Theorem 2 — that task-specific weight
shifts ΔW* have spectral decay that varies across tasks and determines LoRA's
approximation rate.

What we measure:
1. SVD of ΔW* = W_fft - W_0 for each (layer, weight matrix)
2. Spectral decay rate α (fitted from σ_k = C·k^{-α})
3. Spectral tail T(r) = Σ_{k>r} σ_k² for each rank r
4. Variation of α across VTAB task categories

Expected results (from Theorem 2):
- Natural tasks (fine-grained classification): α ≈ 1-2, moderate rank needed
- Structured tasks (counting, spatial): α ≈ 0.5-1, higher rank or VPT needed
- Specialized tasks (medical, satellite): α varies
"""

import torch
import torch.nn as nn
import timm
import numpy as np
import json
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import (ExperimentConfig, setup_device, ensure_dirs, save_results,
                    compute_svd_profile, fit_spectral_decay)


def load_vtab_task(task_name: str, config: ExperimentConfig, split='train'):
    """
    Load a VTAB-1K task. Uses torchvision or HuggingFace datasets.
    Falls back to synthetic data if dataset not available.
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    try:
        from datasets import load_dataset
        # Map task names to HuggingFace dataset names
        hf_map = {
            'cifar100': ('cifar100', None, 'fine_label'),
            'caltech101': ('caltech101', None, 'label'),
            'dtd': ('dtd', None, 'label'),
            'oxford_flowers102': ('nelorth/oxford-flowers', None, 'label'),
            'oxford_iiit_pet': ('timm/oxford-iiit-pet', None, 'label'),
            'svhn': ('svhn', 'cropped_digits', 'label'),
            'sun397': ('tanganke/sun397', None, 'label'),
            'eurosat': ('timm/eurosat-rgb', None, 'label'),
            'patch_camelyon': ('1aurent/PatchCamelyon', None, 'label'),
            'resisc45': ('timm/resisc45', None, 'label'),
            'clevr_count': ('clevr', None, None),
            'dmlab': ('vtab/dmlab', None, 'label'),
            'kitti': ('vtab/kitti', None, 'label'),
        }

        if task_name in hf_map:
            ds_name, ds_config, label_key = hf_map[task_name]
            ds = load_dataset(ds_name, ds_config, split=split, trust_remote_code=True)
            return ds, label_key, transform
    except Exception as e:
        print(f"  Could not load {task_name}: {e}")

    # Fallback: synthetic dataset
    print(f"  Using synthetic data for {task_name}")
    return None, None, transform


class SyntheticVTABDataset(torch.utils.data.Dataset):
    """Synthetic dataset mimicking VTAB-1K structure."""
    def __init__(self, n_samples, n_classes, img_size, task_type='natural'):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.img_size = img_size
        self.task_type = task_type
        self.labels = torch.randint(0, n_classes, (n_samples,))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Generate class-dependent synthetic images
        label = self.labels[idx].item()
        torch.manual_seed(idx)
        img = torch.randn(3, self.img_size, self.img_size) * 0.2
        # Add class-specific signal
        freq = (label + 1) * 2
        t = torch.linspace(0, freq * np.pi, self.img_size)
        pattern = torch.sin(t).unsqueeze(0).unsqueeze(0)
        if self.task_type == 'structured':
            # Spatial structure for counting/spatial tasks
            cx, cy = (label % 5) * self.img_size // 5, (label // 5) * self.img_size // 5
            img[:, max(0,cy):min(self.img_size,cy+40), max(0,cx):min(self.img_size,cx+40)] += 1.0
        else:
            img += pattern * 0.5
        # Normalize
        img = (img - img.mean()) / (img.std() + 1e-6) * 0.224 + 0.456
        return img, label


def full_finetune(model, train_loader, n_classes, config, device,
                  val_loader=None):
    """Run full fine-tuning with cosine LR, warmup, and best-model selection."""
    # Replace classification head
    model.head = nn.Linear(config.embed_dim, n_classes).to(device)
    nn.init.zeros_(model.head.bias)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Cosine schedule with linear warmup
    total_steps = config.epochs * len(train_loader)
    warmup_steps = config.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_state = None
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Validate
        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            val_total, val_count = 0, 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    val_total += criterion(model(batch_x), batch_y).item() * batch_y.shape[0]
                    val_count += batch_y.shape[0]
            val_loss = val_total / max(val_count, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 1 == 0 or epoch == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch+1}/{config.epochs}, "
                  f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {lr_now:.6f}"
                  f"  {'*' if val_loss <= best_val_loss else ''}")

        if patience_counter >= patience and epoch > config.warmup_epochs + 10:
            print(f"    Early stopping at epoch {epoch+1} (best val: {best_val_loss:.4f})")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"    Restored best model (val loss: {best_val_loss:.4f})")

    return model


def extract_weight_shifts(model_pretrained_state, model_finetuned_state,
                          config: ExperimentConfig):
    """
    Extract ΔW = W_fft - W_0 for all relevant weight matrices.
    Returns dict: {layer_matrix_name: delta_W tensor}
    """
    shifts = {}

    for name in model_pretrained_state:
        # Focus on attention projections and FFN
        if any(key in name for key in ['qkv.weight', 'proj.weight',
                                        'mlp.fc1.weight', 'mlp.fc2.weight']):
            W0 = model_pretrained_state[name].float()
            W1 = model_finetuned_state[name].float()
            delta = W1 - W0
            shifts[name] = delta

    return shifts


def load_real_dataset(task_name: str, n_classes: int, config: ExperimentConfig,
                      max_samples: int = None):
    """
    Load a real dataset for VTAB experiments.
    Tries torchvision → HuggingFace → returns None.

    Args:
        max_samples: How many samples to use. None = config.n_train, 0 = ALL.
    """
    from torchvision import transforms
    from torch.utils.data import Subset

    if max_samples is None:
        n_samples = config.n_train
    elif max_samples == 0:
        n_samples = None  # use all
    else:
        n_samples = max_samples

    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def maybe_subsample(ds):
        if n_samples is None or n_samples >= len(ds):
            return ds
        indices = torch.randperm(len(ds))[:n_samples].tolist()
        return Subset(ds, indices)

    # --- torchvision datasets ---
    try:
        import torchvision.datasets as tv_datasets

        tv_loaders = {
            'cifar100': lambda: tv_datasets.CIFAR100(
                root='./data', train=True, download=True, transform=transform),
            'svhn': lambda: tv_datasets.SVHN(
                root='./data', split='train', download=True, transform=transform),
            'dtd': lambda: tv_datasets.DTD(
                root='./data', split='train', download=True, transform=transform),
            'oxford_flowers102': lambda: tv_datasets.Flowers102(
                root='./data', split='train', download=True, transform=transform),
            'eurosat': lambda: tv_datasets.EuroSAT(
                root='./data', download=True, transform=transform),
            # Structured tasks available in torchvision:
            'gtsrb': lambda: tv_datasets.GTSRB(
                root='./data', split='train', download=True, transform=transform),
            'mnist': lambda: tv_datasets.MNIST(
                root='./data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Grayscale(3),  # convert to 3-channel
                    transforms.Resize((config.img_size, config.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
        }

        if task_name in tv_loaders:
            ds = tv_loaders[task_name]()
            print(f"  Loaded {task_name} via torchvision: {len(ds)} samples")
            return maybe_subsample(ds)

    except Exception as e:
        print(f"  torchvision load failed for {task_name}: {e}")

    # --- HuggingFace datasets ---
    try:
        from datasets import load_dataset
        from PIL import Image

        class HFDatasetWrapper(torch.utils.data.Dataset):
            """Wraps a HuggingFace dataset for PyTorch training."""
            def __init__(self, hf_ds, img_key, lbl_key, tfm, n_max,
                         label_fn=None):
                self.ds = hf_ds
                self.img_key = img_key
                self.lbl_key = lbl_key
                self.label_fn = label_fn  # optional label transform
                self.transform = tfm
                self.n = min(n_max, len(hf_ds)) if n_max else len(hf_ds)

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                item = self.ds[idx]
                img = item[self.img_key]
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(np.array(img))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = self.transform(img)
                if self.label_fn:
                    label = self.label_fn(item)
                else:
                    label = item[self.lbl_key]
                return img, label

        # Standard HF datasets (image_key, label_key)
        hf_simple = {
            'cifar100': ('cifar100', None, 'fine_label', 'img'),
            'caltech101': ('caltech101', None, 'label', 'image'),
            'dtd': ('dtd', None, 'label', 'image'),
            'svhn': ('svhn', 'cropped_digits', 'label', 'image'),
            'eurosat': ('timm/eurosat-rgb', None, 'label', 'image'),
            'patch_camelyon': ('1aurent/PatchCamelyon', None, 'label', 'image'),
            'resisc45': ('timm/resisc45', None, 'label', 'image'),
            'sun397': ('tanganke/sun397', None, 'label', 'image'),
            'oxford_iiit_pet': ('timm/oxford-iiit-pet', None, 'label', 'image'),
        }

        if task_name in hf_simple:
            ds_name, ds_config, lbl_key, img_key = hf_simple[task_name]
            ds = load_dataset(ds_name, ds_config, split='train',
                              trust_remote_code=True)
            print(f"  Loaded {task_name} via HuggingFace: {len(ds)} samples")
            return HFDatasetWrapper(ds, img_key, lbl_key, transform,
                                    n_samples)

        # --- Structured tasks with special label extraction ---

        if task_name == 'clevr_count':
            # CLEVR counting: label = number of objects (capped at 10)
            try:
                ds = load_dataset('clevr', split='train',
                                  trust_remote_code=True)
                def count_label(item):
                    # CLEVR objects stored in metadata
                    if 'objects' in item and 'size' in item['objects']:
                        count = len(item['objects']['size'])
                    elif 'question' in item:
                        count = 5  # fallback
                    else:
                        count = min(len(item.get('objects', [])), 10)
                    return min(count, 7)  # bin to 0-7 (8 classes)

                print(f"  Loaded clevr_count via HuggingFace: {len(ds)} samples")
                return HFDatasetWrapper(ds, 'image', None, transform,
                                        n_samples, label_fn=count_label)
            except Exception as e:
                print(f"  CLEVR HF load failed: {e}")

        if task_name == 'smallnorb_azi':
            try:
                ds = load_dataset('smallnorb', split='train',
                                  trust_remote_code=True)
                def azimuth_label(item):
                    return item['label_azimuth']  # 0-17 (18 classes)

                print(f"  Loaded smallnorb via HuggingFace: {len(ds)} samples")
                return HFDatasetWrapper(ds, 'image', None, transform,
                                        n_samples, label_fn=azimuth_label)
            except Exception as e:
                print(f"  SmallNORB HF load failed: {e}")

        if task_name == 'smallnorb_ele':
            try:
                ds = load_dataset('smallnorb', split='train',
                                  trust_remote_code=True)
                def elevation_label(item):
                    return item['label_elevation']  # 0-8 (9 classes)

                print(f"  Loaded smallnorb (elevation) via HuggingFace: {len(ds)} samples")
                return HFDatasetWrapper(ds, 'image', None, transform,
                                        n_samples, label_fn=elevation_label)
            except Exception as e:
                print(f"  SmallNORB HF load failed: {e}")

        if task_name == 'dsprites_loc':
            try:
                ds = load_dataset('dsprites', split='train',
                                  trust_remote_code=True)
                def loc_label(item):
                    # Bin x-position into 16 classes
                    x_pos = item.get('label_x_position',
                                    item.get('value_x_position', 0.5))
                    return min(int(x_pos * 16), 15)

                print(f"  Loaded dsprites via HuggingFace: {len(ds)} samples")
                return HFDatasetWrapper(ds, 'image', None, transform,
                                        n_samples, label_fn=loc_label)
            except Exception as e:
                print(f"  dSprites HF load failed: {e}")

        if task_name == 'dsprites_ori':
            try:
                ds = load_dataset('dsprites', split='train',
                                  trust_remote_code=True)
                def ori_label(item):
                    ori = item.get('label_orientation',
                                  item.get('value_orientation', 0.0))
                    return min(int(ori * 16 / (2 * 3.14159)), 15)

                print(f"  Loaded dsprites (orientation) via HuggingFace: {len(ds)} samples")
                return HFDatasetWrapper(ds, 'image', None, transform,
                                        n_samples, label_fn=ori_label)
            except Exception as e:
                print(f"  dSprites HF load failed: {e}")

        if task_name == 'dmlab':
            try:
                ds = load_dataset('vtab/dmlab', split='train',
                                  trust_remote_code=True)
                print(f"  Loaded dmlab via HuggingFace: {len(ds)} samples")
                return HFDatasetWrapper(ds, 'image', 'label', transform,
                                        n_samples)
            except Exception as e:
                print(f"  DMLab HF load failed: {e}")

    except ImportError:
        print("  HuggingFace datasets not installed. pip install datasets")
    except Exception as e:
        print(f"  HuggingFace load failed for {task_name}: {e}")

    print(f"  WARNING: No real dataset found for {task_name}")
    return None


def run_spectral_analysis(config: ExperimentConfig):
    """Main spectral analysis experiment."""
    print("=" * 70)
    print("EXPERIMENT 1: SPECTRAL PROFILE ANALYSIS")
    print("=" * 70)

    device = setup_device()
    ensure_dirs(config)

    # Load pretrained model
    print("\nLoading pretrained model...")
    try:
        model = timm.create_model(config.model_name, pretrained=True)
    except Exception:
        print("Cannot download pretrained model. Using random init for demo.")
        model = timm.create_model(config.model_name, pretrained=False)

    model = model.to(device)
    pretrained_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    results = {}
    all_alphas = {'natural': [], 'specialized': [], 'structured': []}

    # Select representative tasks (one per subcategory for speed)
    tasks_to_run = ['cifar100', 'dtd', 'svhn', 'eurosat', 'patch_camelyon',
                    'gtsrb', 'clevr_count', 'dsprites_loc', 'smallnorb_azi']

    for task_name in tasks_to_run:
        print(f"\n{'='*50}")
        print(f"Task: {task_name} ({config.task_category(task_name)})")
        print(f"{'='*50}")

        # Reset model to pretrained weights
        model.load_state_dict({k: v.to(device) for k, v in pretrained_state.items()},
                              strict=False)

        # Determine number of classes
        n_classes_map = {
            'cifar100': 100, 'caltech101': 102, 'dtd': 47,
            'oxford_flowers102': 102, 'oxford_iiit_pet': 37,
            'svhn': 10, 'sun397': 397, 'eurosat': 10,
            'patch_camelyon': 2, 'resisc45': 45,
            'clevr_count': 8, 'clevr_dist': 6, 'dmlab': 6,
            'kitti': 4, 'dsprites_loc': 16, 'dsprites_ori': 16,
            'smallnorb_azi': 18, 'smallnorb_ele': 9,
            'diabetic_retinopathy': 5, 'gtsrb': 43,
        }
        n_classes = n_classes_map.get(task_name, 10)
        category = config.task_category(task_name)
        task_type = 'structured' if category == 'structured' else 'natural'

        # Create dataset — use ALL available data for FFT (spectral analysis needs good ΔW*)
        dataset = load_real_dataset(task_name, n_classes, config,
                                    max_samples=config.n_train_fft)
        if dataset is None:
            print(f"  Falling back to synthetic data for {task_name}")
            n_synth = config.n_train_fft if config.n_train_fft > 0 else 5000
            dataset = SyntheticVTABDataset(n_synth, n_classes,
                                           config.img_size, task_type)
        else:
            print(f"  Loaded real dataset: {len(dataset)} samples")

        # Train/val split — use 10% for val (not fixed 200 which is too few for large datasets)
        from torch.utils.data import random_split
        n_total = len(dataset)
        n_val = max(200, n_total // 10)  # at least 200, up to 10% of data
        n_val = min(n_val, 10000)         # cap at 10K to save compute
        n_train = n_total - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                            num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                                num_workers=2)

        # Full fine-tuning
        n_batches = len(loader)
        print(f"  Fine-tuning ({config.epochs} epochs, {n_train} train, {n_val} val)")
        print(f"  Batches/epoch: {n_batches}, img_size: {config.img_size}x{config.img_size}")

        import time as _time
        task_start = _time.time()
        model = full_finetune(model, loader, n_classes, config, device,
                              val_loader=val_loader)
        task_time = _time.time() - task_start
        print(f"  Training completed in {task_time/60:.1f} minutes")

        # Extract weight shifts
        finetuned_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        shifts = extract_weight_shifts(pretrained_state, finetuned_state, config)

        # Spectral analysis
        task_result = {'spectral_profiles': {}, 'decay_rates': {}}
        task_alphas = []

        for name, delta_W in shifts.items():
            sv = compute_svd_profile(delta_W)
            alpha, C = fit_spectral_decay(sv)

            task_result['spectral_profiles'][name] = {
                'singular_values': sv[:50].tolist(),
                'alpha': alpha,
                'C': C,
                'frobenius_norm': float(np.sqrt((sv**2).sum())),
                'nuclear_norm': float(sv.sum()),
                'rank_90pct': int(np.searchsorted(np.cumsum(sv**2) / (sv**2).sum(), 0.9)) + 1,
            }
            task_result['decay_rates'][name] = alpha
            task_alphas.append(alpha)

        # Aggregate statistics
        task_result['mean_alpha'] = float(np.mean(task_alphas))
        task_result['median_alpha'] = float(np.median(task_alphas))
        task_result['std_alpha'] = float(np.std(task_alphas))
        task_result['category'] = category

        # Compute spectral tail T(r) for different ranks
        tail_values = {}
        for r in config.lora_ranks:
            tail = 0.0
            for name, delta_W in shifts.items():
                sv = compute_svd_profile(delta_W)
                tail += (sv[r:] ** 2).sum()
            tail_values[str(r)] = float(tail)
        task_result['spectral_tail'] = tail_values

        results[task_name] = task_result
        all_alphas[category].append(task_result['mean_alpha'])

        print(f"  Mean α = {task_result['mean_alpha']:.3f} ± {task_result['std_alpha']:.3f}")
        print(f"  Spectral tail T(4) = {tail_values['4']:.4f}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: Spectral Decay by Task Category")
    print("=" * 70)
    for cat in ['natural', 'specialized', 'structured']:
        if all_alphas[cat]:
            mean_a = np.mean(all_alphas[cat])
            std_a = np.std(all_alphas[cat])
            print(f"  {cat:12s}: α = {mean_a:.3f} ± {std_a:.3f} (n={len(all_alphas[cat])})")

    # Theorem 2 prediction
    print("\n  Theorem 2 predictions:")
    print("  - Higher α → faster decay → lower LoRA rank needed")
    print("  - Structured tasks should have lower α (need more capacity)")
    print("  - Natural tasks should have higher α (low rank suffices)")

    save_results(results, 'experiment1_spectral.json', config)
    return results


if __name__ == '__main__':
    config = ExperimentConfig(epochs=20)  # Reduced for faster demo
    run_spectral_analysis(config)
