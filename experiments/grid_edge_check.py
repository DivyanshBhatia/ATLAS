"""
Grid Edge Check: Extend LR grid for p=50 where the best LR is at the boundary.

If p=50's best LR is 5e-4 (lowest in grid), try 2e-4 and 1e-4.
If p=50's best LR is 1e-2 (highest in grid), try 2e-2.

This checks whether "capacity hurts" is real or whether p=50 just needs
a lower/higher LR than our grid covers.

Usage:
    cd /content/ATLAS
    python experiments/grid_edge_check.py
    python experiments/grid_edge_check.py --ibot_checkpoint /content/checkpoint_teacher.pth
"""
import sys
sys.path.insert(0, '.')

import argparse
import torch
import torch.nn as nn
import timm
import numpy as np
import json
import os
from copy import deepcopy

from config import ExperimentConfig, setup_device
from exp2_comparison import apply_vpt, train_and_evaluate
from run_all_backbones import TASKS, load_dataset
from torch.utils.data import DataLoader, random_split

SAVE_PATH = 'results/grid_edge_check.json'

# Cases where p=50 best is at grid edge {5e-4 ... 1e-2}
EDGE_CASES = {
    # Low edge (best = 5e-4): try 2e-4, 1e-4
    'low': {
        'extend_lrs': [2e-4, 1e-4],
        'cells': [
            ('CLIP', 'svhn', 0.840),
            ('DINOv2', 'svhn', 0.795),
            ('DINOv1', 'svhn', 0.845),
            ('DINOv1', 'eurosat', 0.940),
            ('MAE', 'svhn', 0.815),      # best at 5e-4
            ('MAE', 'eurosat', 0.950),    # best at 5e-4
        ],
    },
    # High edge (best = 1e-2): try 2e-2
    'high': {
        'extend_lrs': [2e-2],
        'cells': [
            ('MAE', 'cifar100', 0.140),
            ('MAE', 'gtsrb', 0.800),
        ],
    },
}

BACKBONES = {
    'DINOv2': ('vit_base_patch14_dinov2.lvd142m', 0.22),
    'CLIP': ('vit_base_patch16_clip_224.openai', 0.18),
    'DINOv1': ('vit_base_patch16_224.dino', 0.19),
    'MAE': ('vit_base_patch16_224.mae', 1.76),
    'MoCo-v3': (None, 2.31),
    'iBOT': (None, 0.15),
}


def load_model(name, device, ibot_checkpoint=None):
    if name == 'MoCo-v3':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, img_size=224)
        url = 'https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar'
        sd = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        if 'state_dict' in sd:
            sd = {k.replace('module.', '').replace('base_encoder.', ''): v
                  for k, v in sd['state_dict'].items()}
        model.load_state_dict(sd, strict=False)
        return model.to(device)
    elif name == 'iBOT':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, img_size=224)
        candidates = [ibot_checkpoint] if ibot_checkpoint else []
        candidates.extend(['/content/ibot/checkpoint_teacher.pth',
                           '/content/checkpoint_teacher.pth'])
        ckpt_path = next((x for x in candidates if x and os.path.isfile(x)), None)
        if ckpt_path is None:
            raise FileNotFoundError(f"iBOT checkpoint not found")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        sd = checkpoint.get('teacher', checkpoint.get('state_dict', checkpoint))
        cleaned = {}
        for key, value in sd.items():
            new_key = key
            for prefix in ('module.', 'teacher.', 'backbone.'):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            if not new_key.startswith(('head.', 'last_layer.')):
                cleaned[new_key] = value
        model.load_state_dict(cleaned, strict=False)
        return model.to(device)
    else:
        return timm.create_model(BACKBONES[name][0], pretrained=True, img_size=224).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ibot_checkpoint', default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = 100

    all_results = {}
    if args.resume and os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            all_results = json.load(f)

    for edge_type, info in EDGE_CASES.items():
        extend_lrs = info['extend_lrs']
        print(f"\n{'='*60}")
        print(f"  Grid Edge Check: {edge_type} edge")
        print(f"  Testing LRs: {extend_lrs}")
        print(f"{'='*60}")

        for bb_name, task, current_best in info['cells']:
            key = f"{bb_name}_{task}_p50"
            if key in all_results:
                print(f"  {key}: already done"); continue

            print(f"\n  --- {bb_name} x {task} (p=50, current best={current_best}) ---")

            base_model = load_model(bb_name, device, args.ibot_checkpoint)
            config.embed_dim = base_model.embed_dim
            config.num_layers = len(base_model.blocks)
            config.num_heads = base_model.blocks[0].attn.num_heads
            config.head_dim = config.embed_dim // config.num_heads
            config.num_classes = TASKS[task][0]

            lr_results = {}
            for lr in extend_lrs:
                torch.manual_seed(42); np.random.seed(42)
                ds = load_dataset(task, 224, max_samples=1000)
                nv = min(200, len(ds) // 5)
                tds, vds = random_split(ds, [len(ds)-nv, nv],
                    generator=torch.Generator().manual_seed(42))
                tl = DataLoader(tds, batch_size=64, shuffle=True, num_workers=2)
                vl = DataLoader(vds, batch_size=64, shuffle=False, num_workers=2)

                m = deepcopy(base_model)
                m.head = nn.Linear(config.embed_dim, config.num_classes).to(device)
                m = apply_vpt(m, 50, config)
                m = m.to(device)
                config.lr = lr
                acc = train_and_evaluate(m, tl, vl, config, device)
                del m; torch.cuda.empty_cache()

                lr_results[f'{lr}'] = float(acc)
                print(f"    LR={lr:.0e}: {acc:.3f}", end="")
                if acc > current_best:
                    print(f"  *** BETTER than grid best ({current_best:.3f})! ***")
                else:
                    print(f"  (grid best: {current_best:.3f})")

            new_best = max(lr_results.values())
            all_results[key] = {
                'backbone': bb_name,
                'task': task,
                'edge_type': edge_type,
                'grid_best': current_best,
                'extended_results': lr_results,
                'new_best': float(new_best),
                'improved': new_best > current_best,
                'improvement': float(new_best - current_best),
            }

            del base_model; torch.cuda.empty_cache()

            os.makedirs('results', exist_ok=True)
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("GRID EDGE CHECK SUMMARY")
    print(f"{'='*60}")
    improved = 0
    for key, data in all_results.items():
        status = "IMPROVED" if data['improved'] else "no change"
        print(f"  {data['backbone']:>8s} x {data['task']:<10s}: "
              f"grid={data['grid_best']:.3f} → extended={data['new_best']:.3f} "
              f"({data['improvement']:+.3f}) {status}")
        if data['improved']:
            improved += 1

    print(f"\n  {improved}/{len(all_results)} cells improved beyond grid.")
    if improved == 0:
        print("  Grid is sufficient — capacity degradation is real.")
    else:
        print(f"  {improved} cells may have understated p=50 accuracy.")


if __name__ == '__main__':
    main()
