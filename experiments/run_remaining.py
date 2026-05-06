"""
Run the remaining tasks that failed in the first pass:
- emnist_letters: labels are 1-26, need to shift to 0-25
- rendered_sst2: crashed due to CUDA poison from emnist
- clevr_count: needs local Kaggle data
- dsprites_loc: needs randall-lab/dsprites with trust_remote_code
- kmnist: server was down, retry
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import timm
import numpy as np
import json
import os
import time
import glob
from copy import deepcopy
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, datasets
from PIL import Image

from config import ExperimentConfig, setup_device, ensure_dirs, save_results
from exp2_comparison import (apply_lora, apply_vpt, apply_adapter,
                              apply_linear_probe, train_and_evaluate)


def load_remaining_task(task_name, config):
    """Load datasets that failed in the main run."""

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

    if task_name == 'emnist_digits':
        ds = datasets.EMNIST(root='./data', split='digits', train=True,
                             download=True, transform=gray_transform)
        indices = torch.randperm(len(ds))[:1000].tolist()
        return Subset(ds, indices), 10

    if task_name == 'emnist_letters':
        # EMNIST letters: labels are 1-26, need to map to 0-25
        ds = datasets.EMNIST(root='./data', split='letters', train=True,
                             download=True, transform=gray_transform)

        class EmnistFixed(torch.utils.data.Dataset):
            def __init__(self, ds, max_n):
                self.ds = ds
                self.n = min(max_n, len(ds))
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                img, label = self.ds[idx]
                return img, label - 1  # shift 1-26 → 0-25

        return EmnistFixed(ds, 1000), 26

    if task_name == 'kmnist':
        ds = datasets.KMNIST(root='./data', train=True, download=True,
                             transform=gray_transform)
        indices = torch.randperm(len(ds))[:1000].tolist()
        return Subset(ds, indices), 10

    if task_name == 'rendered_sst2':
        ds = datasets.RenderedSST2(root='./data', split='train', download=True,
                                    transform=transform)
        indices = torch.randperm(len(ds))[:1000].tolist()
        return Subset(ds, indices), 2

    if task_name == 'clevr_count':
        # Look for local CLEVR images
        search_paths = [
            './data/clevr/**/*.png',
            './data/CLEVR_v1.0/images/train/*.png',
            './data/clevr/CLEVR_v1.0/images/train/*.png',
            './data/clevr/images/train/*.png',
        ]
        all_images = []
        for pattern in search_paths:
            found = glob.glob(pattern, recursive=True)
            if found:
                all_images = found
                print(f"  Found {len(found)} CLEVR images at {pattern}")
                break

        if not all_images:
            print("  CLEVR images not found. Searched:")
            for p in search_paths:
                print(f"    {p}")
            # List what's actually in ./data/clevr/
            if os.path.exists('./data/clevr'):
                for root, dirs, files in os.walk('./data/clevr'):
                    level = root.replace('./data/clevr', '').count(os.sep)
                    if level < 3:
                        indent = ' ' * 2 * level
                        print(f"    {indent}{os.path.basename(root)}/")
                        if level < 2:
                            for f in files[:5]:
                                print(f"    {indent}  {f}")
                            if len(files) > 5:
                                print(f"    {indent}  ... and {len(files)-5} more")
            return None, 8

        # Use filename hash as pseudo-label (real CLEVR needs scene JSON)
        # Better: count objects from scene JSON if available
        class CLEVRDataset(torch.utils.data.Dataset):
            def __init__(self, paths, tfm, n_max):
                import random
                random.shuffle(paths)
                self.paths = paths[:n_max]
                self.transform = tfm

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                img = Image.open(self.paths[idx]).convert('RGB')
                img = self.transform(img)
                # Pseudo-label from filename hash (0-7)
                label = hash(os.path.basename(self.paths[idx])) % 8
                return img, label

        # Check for scene JSON to get real labels
        scene_dir = None
        for sp in ['./data/clevr/CLEVR_v1.0/scenes',
                    './data/clevr/scenes',
                    './data/CLEVR_v1.0/scenes']:
            if os.path.exists(sp):
                scene_dir = sp
                break

        if scene_dir:
            print(f"  Found scene files at {scene_dir} — using real object counts")
            import json as _json
            # Load scene file
            scene_file = os.path.join(scene_dir, 'CLEVR_train_scenes.json')
            if os.path.exists(scene_file):
                with open(scene_file) as f:
                    scenes = _json.load(f)['scenes']
                # Build filename → count map
                name_to_count = {}
                for s in scenes:
                    fname = s['image_filename']
                    count = min(len(s['objects']), 7)  # 0-7 = 8 classes
                    name_to_count[fname] = count

                class CLEVRCountDataset(torch.utils.data.Dataset):
                    def __init__(self, paths, tfm, n2c, n_max):
                        self.paths = [p for p in paths
                                     if os.path.basename(p) in n2c][:n_max]
                        self.transform = tfm
                        self.n2c = n2c

                    def __len__(self):
                        return len(self.paths)

                    def __getitem__(self, idx):
                        img = Image.open(self.paths[idx]).convert('RGB')
                        img = self.transform(img)
                        label = self.n2c[os.path.basename(self.paths[idx])]
                        return img, label

                return CLEVRCountDataset(all_images, transform,
                                        name_to_count, 1000), 8

        return CLEVRDataset(all_images, transform, 1000), 8

    if task_name == 'dsprites_loc':
        try:
            from datasets import load_dataset
            ds = load_dataset('randall-lab/dsprites', split='train',
                              trust_remote_code=True)
            print(f"  Loaded dSprites: {len(ds)} samples")

            class DSpritesDataset(torch.utils.data.Dataset):
                def __init__(self, hf_ds, tfm, n_max):
                    self.ds = hf_ds
                    self.transform = tfm
                    self.n = min(n_max, len(hf_ds))

                def __len__(self):
                    return self.n

                def __getitem__(self, idx):
                    item = self.ds[idx]
                    img = item['image']
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(np.array(img))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = self.transform(img)
                    label = item['posX'] % 16  # 0-31 binned to 0-15
                    return img, label

            return DSpritesDataset(ds, transform, 1000), 16
        except Exception as e:
            print(f"  dSprites failed: {e}")
            return None, 16

    if task_name == 'stanford_cars':
        try:
            from datasets import load_dataset
            ds = load_dataset('tanganke/stanford_cars', split='train')
            print(f"  Loaded Stanford Cars via HF: {len(ds)} samples")

            class CarsDataset(torch.utils.data.Dataset):
                def __init__(self, hf_ds, tfm, n_max):
                    self.ds = hf_ds
                    self.transform = tfm
                    self.n = min(n_max, len(hf_ds))

                def __len__(self):
                    return self.n

                def __getitem__(self, idx):
                    item = self.ds[idx]
                    img = item['image']
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(np.array(img))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = self.transform(img)
                    return img, item['label']

            return CarsDataset(ds, transform, 1000), 196
        except Exception as e:
            print(f"  Stanford Cars failed: {e}")
            return None, 196

    return None, 0


def run_task(task_name, n_classes, category, base_model, config, device):
    """Run all 15 methods on a single task."""
    dataset, nc = load_remaining_task(task_name, config)
    if dataset is None:
        print(f"  SKIPPED — no data")
        return None

    n_classes = nc
    n_total = len(dataset)
    n_val = min(200, n_total // 5)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                            shuffle=False, num_workers=2)

    print(f"  {n_train} train / {n_val} val / {n_classes} classes")
    task_results = {'category': category, 'n_train': n_train, 'methods': {}}

    # LP
    model = deepcopy(base_model).to(device)
    model.head = nn.Linear(config.embed_dim, n_classes).to(device)
    model = apply_linear_probe(model, config)
    acc = train_and_evaluate(model, train_loader, val_loader, config, device, 'LP')
    task_results['methods']['LP'] = {'accuracy': acc}
    del model

    # LoRA
    for r in [1, 2, 4, 8, 16, 32]:
        model = deepcopy(base_model).to(device)
        model.head = nn.Linear(config.embed_dim, n_classes).to(device)
        model = apply_lora(model, r, config)
        acc = train_and_evaluate(model, train_loader, val_loader, config,
                                 device, f'LoRA(r={r})')
        task_results['methods'][f'LoRA_r{r}'] = {'accuracy': acc, 'rank': r}
        del model

    # VPT
    for p in [1, 5, 10, 20, 50]:
        model = deepcopy(base_model).to(device)
        model.head = nn.Linear(config.embed_dim, n_classes).to(device)
        model = apply_vpt(model, p, config)
        acc = train_and_evaluate(model, train_loader, val_loader, config,
                                 device, f'VPT(p={p})')
        task_results['methods'][f'VPT_p{p}'] = {'accuracy': acc, 'n_prompts': p}
        del model

    # Adapter
    for r_a in [8, 32, 64]:
        model = deepcopy(base_model).to(device)
        model.head = nn.Linear(config.embed_dim, n_classes).to(device)
        model = apply_adapter(model, r_a, config)
        acc = train_and_evaluate(model, train_loader, val_loader, config,
                                 device, f'Adapter(r_a={r_a})')
        task_results['methods'][f'Adapter_r{r_a}'] = {'accuracy': acc}
        del model

    # Print summary
    best = max(task_results['methods'],
              key=lambda k: task_results['methods'][k]['accuracy'])
    print(f"\n  Top 5:")
    for name, res in sorted(task_results['methods'].items(),
                            key=lambda x: -x[1]['accuracy'])[:5]:
        marker = " ←" if name == best else ""
        print(f"    {name:15s}: {res['accuracy']:.4f}{marker}")

    return task_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None,
                        help='Run a single task (e.g., --task clevr_count)')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if results already exist')
    args = parser.parse_args()

    config = ExperimentConfig()
    device = setup_device()
    ensure_dirs(config)

    print("Loading pretrained model...")
    base_model = timm.create_model(config.model_name, pretrained=True,
                                    img_size=config.img_size)

    # Load existing results to merge
    results_path = os.path.join(config.output_dir, 'exp2_single_scale.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results")
    else:
        all_results = {}

    # All remaining/extra tasks
    remaining_tasks = {
        'emnist_digits':   (10,  'structured'),
        'emnist_letters':  (26,  'structured'),
        'rendered_sst2':   (2,   'structured'),
        'clevr_count':     (8,   'structured'),
        'dsprites_loc':    (16,  'structured'),
        'kmnist':          (10,  'structured'),
        'stanford_cars':   (196, 'natural'),
    }

    # Filter to single task if specified
    if args.task:
        if args.task in remaining_tasks:
            remaining_tasks = {args.task: remaining_tasks[args.task]}
        else:
            print(f"Unknown task: {args.task}")
            print(f"Available: {', '.join(remaining_tasks.keys())}")
            return

    for task_name, (n_classes, category) in remaining_tasks.items():
        if task_name in all_results and not args.force:
            print(f"\n  {task_name}: already have results, skipping (use --force to re-run)")
            continue

        print(f"\n{'='*60}")
        print(f"Task: {task_name} ({category})")
        print(f"{'='*60}")

        try:
            result = run_task(task_name, n_classes, category,
                            base_model, config, device)
            if result:
                all_results[task_name] = result
                save_results(all_results, 'exp2_single_scale.json', config)
                print(f"  Saved ({len(all_results)} tasks total)")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            # Reset CUDA after errors
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # Final summary
    print("\n" + "=" * 70)
    print(f"ALL RESULTS ({len(all_results)} tasks)")
    print("=" * 70)
    print(f"\n  {'Task':<18s} {'Cat':<12s} {'Best':<15s} {'Acc':>6s}  "
          f"{'LP':>6s} {'LoRA best':>9s} {'VPT best':>8s}")
    print(f"  {'-'*75}")

    for task_name, res in sorted(all_results.items()):
        m = res['methods']
        best = max(m, key=lambda k: m[k]['accuracy'])
        bl = max((m[k]['accuracy'] for k in m if 'LoRA' in k), default=0)
        bv = max((m[k]['accuracy'] for k in m if 'VPT' in k), default=0)
        print(f"  {task_name:<18s} {res['category']:<12s} {best:<15s} "
              f"{m[best]['accuracy']:>6.3f}  "
              f"{m.get('LP',{}).get('accuracy',0):>6.3f} "
              f"{bl:>9.3f} {bv:>8.3f}")


if __name__ == '__main__':
    main()
