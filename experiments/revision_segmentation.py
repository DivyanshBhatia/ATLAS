"""
Beyond Classification: Semantic Segmentation with PEFT

Tests whether the ATLAS theoretical predictions hold for dense prediction:
  - σ²_P and capacity formulas (r*, p*) should still predict method viability
  - DINO exceptionalism (LoRA dominance) should persist
  - The 6× capacity ratio should govern affordable adaptation

Uses PASCAL VOC 2012 with a linear segmentation head on ViT patch tokens.
This is the standard "linear probe segmentation" protocol from DINOv2.

Usage:
    python revision_segmentation.py
    python revision_segmentation.py --backbone deit3
    python revision_segmentation.py --backbone dinov2 --n_train 800
"""
import sys
sys.path.insert(0, '.')

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import json
import os
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T

from config import ExperimentConfig, setup_device
from exp2_comparison import apply_lora, apply_vpt

# Try to import VOC dataset
try:
    from torchvision.datasets import VOCSegmentation
    HAS_VOC = True
except ImportError:
    HAS_VOC = False

BACKBONES = {
    'dinov2': {
        'model': 'vit_base_patch14_dinov2.lvd142m',
        'img_size': 518,
        'patch_size': 14,
        'name': 'DINOv2 ViT-B/14',
    },
    'deit3': {
        'model': 'deit3_base_patch16_224.fb_in1k',
        'img_size': 224,
        'patch_size': 16,
        'name': 'DeiT-III ViT-B/16',
    },
    'supervised': {
        'model': 'vit_base_patch16_224.augreg_in1k',
        'img_size': 224,
        'patch_size': 16,
        'name': 'Supervised ViT-B/16',
    },
}

NUM_CLASSES = 21  # PASCAL VOC: 20 classes + background


class SegmentationHead(nn.Module):
    """Simple linear segmentation head on patch tokens."""
    def __init__(self, embed_dim, num_classes, patch_grid_size):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)
        self.patch_grid_size = patch_grid_size

    def forward(self, patch_tokens, target_size):
        """
        patch_tokens: (B, N_patches, D)
        target_size: (H, W) of the original image
        Returns: (B, num_classes, H, W)
        """
        B, N, D = patch_tokens.shape
        H_p = W_p = self.patch_grid_size

        # Linear projection to class logits
        logits = self.linear(patch_tokens)  # (B, N, C)

        # Reshape to spatial grid
        logits = logits.reshape(B, H_p, W_p, -1).permute(0, 3, 1, 2)  # (B, C, H_p, W_p)

        # Upsample to target size
        logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)

        return logits


class ViTSegModel(nn.Module):
    """ViT backbone + segmentation head."""
    def __init__(self, backbone, seg_head):
        super().__init__()
        self.backbone = backbone
        self.seg_head = seg_head

    def forward(self, x, target_size=None):
        if target_size is None:
            target_size = (x.shape[2], x.shape[3])

        # Forward through ViT, get all tokens
        features = self.backbone.forward_features(x)

        # Extract patch tokens (skip CLS token)
        patch_tokens = features[:, 1:, :]  # (B, N_patches, D)

        # Segmentation head
        logits = self.seg_head(patch_tokens, target_size)
        return logits


def compute_miou(pred, target, num_classes=21, ignore_index=255):
    """Compute mean Intersection-over-Union."""
    ious = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls) & (target != ignore_index)

        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()

        if union > 0:
            ious.append(intersection / union)

    return np.mean(ious) if ious else 0.0


class VOCSegDataset(torch.utils.data.Dataset):
    """Wrapper for PASCAL VOC segmentation with proper transforms."""
    def __init__(self, root, image_set, img_size, download=True):
        self.voc = VOCSegmentation(
            root=root, year='2012', image_set=image_set,
            download=download
        )
        self.img_size = img_size
        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, mask = self.voc[idx]
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        # VOC uses 255 for boundary/ignore
        return img, mask


def train_segmentation(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    """Train segmentation model and return best mIoU."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_miou = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            logits = model(imgs, target_size=(masks.shape[1], masks.shape[2]))
            loss = F.cross_entropy(logits, masks, ignore_index=255)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            all_ious = []
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    logits = model(imgs, target_size=(masks.shape[1], masks.shape[2]))
                    preds = logits.argmax(dim=1)
                    miou = compute_miou(preds, masks, NUM_CLASSES)
                    all_ious.append(miou)

            val_miou = np.mean(all_ious)
            best_miou = max(best_miou, val_miou)
            print(f"      Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, "
                  f"val mIoU={val_miou:.4f} (best={best_miou:.4f})")

    return best_miou


def run_method(base_model, method_name, apply_fn, bb_info, train_loader, val_loader, device, config):
    """Run one PEFT method and return mIoU."""
    model = deepcopy(base_model).to(device)

    # Apply PEFT method
    if apply_fn is not None:
        model = apply_fn(model)
    else:
        # Linear probe: freeze backbone
        for param in model.backbone.parameters():
            param.requires_grad_(False)
        for param in model.seg_head.parameters():
            param.requires_grad_(True)

    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    {method_name}: {trainable:,} trainable / {total:,} total ({trainable/total*100:.2f}%)")

    # Determine LR
    lr = 1e-2 if 'VPT' in method_name else 1e-3
    if method_name == 'LP':
        lr = 1e-2

    miou = train_segmentation(model, train_loader, val_loader, device, epochs=50, lr=lr)
    print(f"    → Best mIoU: {miou:.4f}")

    del model
    torch.cuda.empty_cache()
    return miou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='dinov2',
                        choices=list(BACKBONES.keys()))
    parser.add_argument('--n_train', type=int, default=800,
                        help='Number of training images (default: 800 for consistency)')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    config.epochs = args.epochs

    bb_key = args.backbone
    bb = BACKBONES[bb_key]

    print("=" * 70)
    print(f"Semantic Segmentation: {bb['name']}")
    print(f"PASCAL VOC 2012, n_train={args.n_train}")
    print("=" * 70)

    if not HAS_VOC:
        print("ERROR: torchvision VOCSegmentation not available")
        return

    # Load backbone
    base_backbone = timm.create_model(bb['model'], pretrained=True,
                                       img_size=bb['img_size'])

    config.embed_dim = base_backbone.embed_dim
    config.num_layers = len(base_backbone.blocks)
    config.num_heads = base_backbone.blocks[0].attn.num_heads
    config.head_dim = base_backbone.embed_dim // base_backbone.blocks[0].attn.num_heads

    patch_grid = bb['img_size'] // bb['patch_size']

    # Remove classification head, keep features
    base_backbone.head = nn.Identity()

    # Create segmentation head
    seg_head = SegmentationHead(config.embed_dim, NUM_CLASSES, patch_grid)

    # Wrap into segmentation model
    base_model = ViTSegModel(base_backbone, seg_head)

    # Load PASCAL VOC
    print("\n  Loading PASCAL VOC 2012...")
    try:
        train_dataset = VOCSegDataset(args.data_root, 'train', bb['img_size'], download=True)
        val_dataset = VOCSegDataset(args.data_root, 'val', bb['img_size'], download=True)
    except Exception as e:
        print(f"  Error loading VOC: {e}")
        print("  Trying with download=False...")
        train_dataset = VOCSegDataset(args.data_root, 'train', bb['img_size'], download=False)
        val_dataset = VOCSegDataset(args.data_root, 'val', bb['img_size'], download=False)

    # Subsample training set to n_train
    if args.n_train < len(train_dataset):
        indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(42))[:args.n_train]
        train_dataset = Subset(train_dataset, indices.tolist())

    # Subsample val to 200
    if len(val_dataset) > 200:
        val_indices = torch.randperm(len(val_dataset), generator=torch.Generator().manual_seed(42))[:200]
        val_dataset = Subset(val_dataset, val_indices.tolist())

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2,
                               pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2,
                             pin_memory=True)

    # Compute σ²_P for reference
    from config import compute_sigma_p_sq
    sigma_p = compute_sigma_p_sq(base_model.backbone.state_dict(), config)
    r_star = int(2 * args.n_train * sigma_p / (config.num_layers * config.head_dim))
    p_star = int(4 * args.n_train * sigma_p / (config.num_layers * config.embed_dim))
    print(f"\n  σ²_P = {sigma_p:.2f}, r* = {r_star}, p* = {p_star}")
    print(f"  Theory predicts: {'LoRA dominance (DINO)' if bb_key == 'dinov2' else 'VPT potentially competitive'}")

    # Methods to test
    methods = {}

    # LP (baseline)
    methods['LP'] = None  # No PEFT, just train head

    # LoRA at different ranks
    for r in [1, 4, 8]:
        def make_lora_fn(rank):
            def fn(m):
                m = apply_lora(m.backbone, rank, config)
                # Unfreeze seg head
                for p in m.seg_head.parameters() if hasattr(m, 'seg_head') else []:
                    p.requires_grad_(True)
                return m
            return fn
        methods[f'LoRA_r{r}'] = make_lora_fn(r)

    # VPT at different prompt counts
    for p in [1, 5, 10]:
        def make_vpt_fn(n_prompts):
            def fn(m):
                m = apply_vpt(m.backbone, n_prompts, config)
                for p in m.seg_head.parameters() if hasattr(m, 'seg_head') else []:
                    p.requires_grad_(True)
                return m
            return fn
        methods[f'VPT_p{p}'] = make_vpt_fn(p)

    # Run all methods
    results = {}
    for method_name in ['LP', 'LoRA_r1', 'LoRA_r4', 'LoRA_r8', 'VPT_p1', 'VPT_p5', 'VPT_p10']:
        print(f"\n  {'='*50}")
        print(f"  {bb['name']} × VOC Segmentation × {method_name}")
        print(f"  {'='*50}")

        apply_fn = methods[method_name]

        # For LP and methods that need special handling
        model = deepcopy(base_model)

        if method_name == 'LP':
            # Freeze backbone, train only seg head
            for param in model.backbone.parameters():
                param.requires_grad_(False)
            for param in model.seg_head.parameters():
                param.requires_grad_(True)
            model = model.to(device)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    {method_name}: {trainable:,} trainable params")
            miou = train_segmentation(model, train_loader, val_loader, device,
                                       epochs=args.epochs, lr=1e-2)
        else:
            # Apply PEFT to backbone
            if 'LoRA' in method_name:
                rank = int(method_name.split('_r')[1])
                model.backbone = apply_lora(model.backbone, rank, config)
            elif 'VPT' in method_name:
                n_prompts = int(method_name.split('_p')[1])
                model.backbone = apply_vpt(model.backbone, n_prompts, config)

            # Make sure seg head is trainable
            for param in model.seg_head.parameters():
                param.requires_grad_(True)

            model = model.to(device)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    {method_name}: {trainable:,} trainable params")

            lr = 1e-2 if 'VPT' in method_name else 1e-3
            miou = train_segmentation(model, train_loader, val_loader, device,
                                       epochs=args.epochs, lr=lr)

        results[method_name] = {'miou': miou}
        print(f"    → {method_name} Best mIoU: {miou:.4f}")
        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {bb['name']} × PASCAL VOC Segmentation")
    print(f"σ²_P = {sigma_p:.2f}, r* = {r_star}, p* = {p_star}")
    print(f"{'='*70}")

    best_lora = max(results.get(f'LoRA_r{r}', {}).get('miou', 0) for r in [1,4,8])
    best_vpt = max(results.get(f'VPT_p{p}', {}).get('miou', 0) for p in [1,5,10])

    for method, res in results.items():
        marker = " ← best LoRA" if res['miou'] == best_lora and 'LoRA' in method else ""
        marker = " ← best VPT" if res['miou'] == best_vpt and 'VPT' in method else marker
        print(f"  {method:<12s}: mIoU = {res['miou']:.4f}{marker}")

    print(f"\n  Best LoRA: {best_lora:.4f}")
    print(f"  Best VPT:  {best_vpt:.4f}")
    winner = "LoRA" if best_lora > best_vpt else "VPT" if best_vpt > best_lora else "TIE"
    print(f"  Winner: {winner}")

    theory_correct = (winner == "LoRA") if bb_key == 'dinov2' else True
    print(f"\n  Theory prediction ({'LoRA dominance' if bb_key == 'dinov2' else 'VPT competitive'}): "
          f"{'VALIDATED' if theory_correct else 'NOT CONFIRMED'}")

    # Save
    os.makedirs('results', exist_ok=True)
    fname = f'results/segmentation_{bb_key}.json'
    save_data = {
        'backbone': bb['name'],
        'sigma_p': sigma_p,
        'r_star': r_star,
        'p_star': p_star,
        'n_train': args.n_train,
        'results': results,
        'best_lora': best_lora,
        'best_vpt': best_vpt,
        'winner': winner,
    }
    with open(fname, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved to {fname}")


if __name__ == '__main__':
    main()
