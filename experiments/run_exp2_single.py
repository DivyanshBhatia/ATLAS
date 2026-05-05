"""
Quick single-scale PEFT comparison — runs parallel to the multi-scale cifar100.
Skips cifar100 (already running). Uses n=1000.
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
from copy import deepcopy
from torch.utils.data import DataLoader, random_split

from config import ExperimentConfig, setup_device, ensure_dirs, save_results
from exp1_spectral import load_real_dataset
from exp2_comparison import (apply_lora, apply_vpt, apply_adapter,
                              apply_linear_probe, train_and_evaluate)


def main():
    config = ExperimentConfig()
    device = setup_device()
    ensure_dirs(config)

    print("Loading pretrained model...")
    base_model = timm.create_model(config.model_name, pretrained=True,
                                    img_size=config.img_size)

    # All tasks EXCEPT cifar100 (running multi-scale separately)
    tasks = {
        'cifar10': (10, 'natural'),
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
        'kmnist': (10, 'structured'),
        'emnist_letters': (26, 'structured'),
        'rendered_sst2': (2, 'structured'),
        'clevr_count': (8, 'structured'),
        'dsprites_loc': (16, 'structured'),
    }

    n_train_target = 1000
    all_results = {}

    for task_name, (n_classes, category) in tasks.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name} ({category})")
        print(f"{'='*60}")
        task_start = time.time()

        try:
            dataset = load_real_dataset(task_name, n_classes, config,
                                        max_samples=n_train_target)
            if dataset is None:
                print(f"  SKIPPED — no real dataset")
                continue

            n_total = len(dataset)
            n_val = min(200, n_total // 5)
            n_train = n_total - n_val
            train_ds, val_ds = random_split(dataset, [n_train, n_val])

            train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                                      shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                                    shuffle=False, num_workers=2)

            print(f"  {n_train} train / {n_val} val")
            task_results = {'category': category, 'n_train': n_train, 'methods': {}}

            # LP
            model = deepcopy(base_model).to(device)
            model.head = nn.Linear(config.embed_dim, n_classes).to(device)
            model = apply_linear_probe(model, config)
            acc = train_and_evaluate(model, train_loader, val_loader, config,
                                     device, 'LP')
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

            # Print task summary
            elapsed = time.time() - task_start
            best = max(task_results['methods'],
                      key=lambda k: task_results['methods'][k]['accuracy'])
            print(f"\n  Results ({elapsed:.0f}s):")
            for name, res in sorted(task_results['methods'].items(),
                                    key=lambda x: -x[1]['accuracy'])[:5]:
                marker = " ←" if name == best else ""
                print(f"    {name:15s}: {res['accuracy']:.4f}{marker}")

            all_results[task_name] = task_results

            # Save after each task (so we don't lose progress)
            save_results(all_results, 'exp2_single_scale.json', config)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    print("\n" + "=" * 70)
    print("SINGLE-SCALE RESULTS (n=1000)")
    print("=" * 70)
    print(f"\n  {'Task':<18s} {'Cat':<12s} {'Best':<15s} {'Acc':>6s}  "
          f"{'LP':>6s} {'LoRA4':>6s} {'VPT10':>6s}")
    print(f"  {'-'*75}")

    for task_name, res in all_results.items():
        m = res['methods']
        best = max(m, key=lambda k: m[k]['accuracy'])
        print(f"  {task_name:<18s} {res['category']:<12s} {best:<15s} "
              f"{m[best]['accuracy']:>6.3f}  "
              f"{m.get('LP',{}).get('accuracy',0):>6.3f} "
              f"{m.get('LoRA_r4',{}).get('accuracy',0):>6.3f} "
              f"{m.get('VPT_p10',{}).get('accuracy',0):>6.3f}")

    # LoRA vs VPT
    print(f"\n  LoRA vs VPT:")
    for task_name, res in all_results.items():
        m = res['methods']
        bl = max((m[k]['accuracy'] for k in m if 'LoRA' in k), default=0)
        bv = max((m[k]['accuracy'] for k in m if 'VPT' in k), default=0)
        w = "LoRA" if bl > bv + 0.01 else "VPT" if bv > bl + 0.01 else "TIE"
        print(f"    {task_name:<18s} LoRA={bl:.3f} VPT={bv:.3f} → {w}")

    # Cross-validate with exp5
    exp5_path = os.path.join(config.output_dir, 'task_structure_analysis.json')
    if os.path.exists(exp5_path):
        with open(exp5_path) as f:
            exp5 = json.load(f)

        print(f"\n  Cross-validation with Exp5 predictions:")
        correct, total = 0, 0
        for task_name, res in all_results.items():
            if task_name not in exp5:
                continue
            e5 = exp5[task_name]
            gap = e5.get('feature_gap', 0)
            avar = e5.get('attention_class_variance_ratio', 0)

            # Prediction
            if gap < 0.10:
                pred = "LP"
            elif avar > 0.3 and gap > 0.2:
                pred = "VPT"
            elif avar > 0.2:
                pred = "VPT/LoRA"
            else:
                pred = "LoRA"

            # Actual
            m = res['methods']
            blp = m.get('LP', {}).get('accuracy', 0)
            bl = max((m[k]['accuracy'] for k in m if 'LoRA' in k), default=0)
            bv = max((m[k]['accuracy'] for k in m if 'VPT' in k), default=0)
            best_all = max(blp, bl, bv)

            if blp >= best_all - 0.02:
                actual = "LP"
            elif bv > bl + 0.01:
                actual = "VPT"
            elif bl > bv + 0.01:
                actual = "LoRA"
            else:
                actual = "TIE"

            match = (pred == "LP" and actual == "LP") or \
                    (pred == "LoRA" and actual in ("LoRA", "TIE")) or \
                    (pred == "VPT" and actual in ("VPT", "TIE")) or \
                    (pred == "VPT/LoRA" and actual in ("VPT", "LoRA", "TIE"))
            correct += match
            total += 1
            sym = "✓" if match else "✗"
            print(f"    {task_name:<18s} gap={gap:.2f} avar={avar:.3f} "
                  f"pred={pred:<10s} actual={actual:<6s} {sym}")

        print(f"\n  Accuracy: {correct}/{total} = {correct/max(total,1):.0%}")


if __name__ == '__main__':
    main()
