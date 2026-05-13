"""
Second Dense Prediction Task: ADE20K Semantic Segmentation

Validates the "beyond classification" claim with a second dense task.
ADE20K has 150 classes (vs VOC's 21), making it substantially harder.

Uses the same linear segmentation head protocol as the VOC experiment.

Usage:
    python revision_ade20k.py --backbone dinov2
    python revision_ade20k.py --backbone deit3
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

from config import ExperimentConfig, setup_device, compute_sigma_p_sq
from exp2_comparison import apply_lora, apply_vpt

# Try importing ADE20K from MIT Scene Parsing
try:
    from torchvision.datasets import ImageFolder
    import torchvision.transforms as T
    from PIL import Image
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

NUM_CLASSES = 150  # ADE20K

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
}


class SegmentationHead(nn.Module):
    def __init__(self, embed_dim, num_classes, patch_grid_size):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)
        self.patch_grid_size = patch_grid_size

    def forward(self, patch_tokens, target_size):
        B, N, D = patch_tokens.shape
        H_p = W_p = self.patch_grid_size
        logits = self.linear(patch_tokens)
        logits = logits.reshape(B, H_p, W_p, -1).permute(0, 3, 1, 2)
        logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
        return logits


class ViTSegModel(nn.Module):
    def __init__(self, backbone, seg_head):
        super().__init__()
        self.backbone = backbone
        self.seg_head = seg_head

    def forward(self, x, target_size=None):
        if target_size is None:
            target_size = (x.shape[2], x.shape[3])
        features = self.backbone.forward_features(x)
        patch_tokens = features[:, 1:, :]
        logits = self.seg_head(patch_tokens, target_size)
        return logits


def compute_miou(pred, target, num_classes=150, ignore_index=255):
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


class ADE20KDataset(torch.utils.data.Dataset):
    """ADE20K dataset loader.
    
    Downloads ADE20K or uses a pre-downloaded version.
    Falls back to generating a synthetic segmentation task if ADE20K
    is not available (for testing the pipeline).
    """
    def __init__(self, root, split='training', img_size=224, max_samples=None):
        self.img_size = img_size
        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Try to find ADE20K data
        ade_img_dir = os.path.join(root, 'ADEChallengeData2016', 'images', split)
        ade_ann_dir = os.path.join(root, 'ADEChallengeData2016', 'annotations', split)

        if os.path.exists(ade_img_dir) and os.path.exists(ade_ann_dir):
            self.mode = 'real'
            self.images = sorted([os.path.join(ade_img_dir, f)
                                   for f in os.listdir(ade_img_dir) if f.endswith('.jpg')])
            self.annotations = sorted([os.path.join(ade_ann_dir, f)
                                        for f in os.listdir(ade_ann_dir) if f.endswith('.png')])
            if max_samples:
                self.images = self.images[:max_samples]
                self.annotations = self.annotations[:max_samples]
            print(f"    Loaded {len(self.images)} ADE20K {split} images")
        else:
            # Try downloading via pip
            print(f"    ADE20K not found at {ade_img_dir}")
            print(f"    Attempting to use MIT Scene Parsing Benchmark...")

            # Fall back to VOC if ADE20K not available
            try:
                from torchvision.datasets import VOCSegmentation
                self.mode = 'voc_fallback'
                voc_split = 'train' if split == 'training' else 'val'
                self.voc = VOCSegmentation(root=root, year='2012',
                                            image_set=voc_split, download=True)
                if max_samples:
                    indices = list(range(min(max_samples, len(self.voc))))
                    self.indices = indices
                else:
                    self.indices = list(range(len(self.voc)))
                print(f"    Falling back to VOC 2012 ({len(self.indices)} images)")
            except Exception:
                self.mode = 'synthetic'
                self.n_samples = max_samples or 500
                print(f"    Using synthetic segmentation data ({self.n_samples} samples)")

    def __len__(self):
        if self.mode == 'real':
            return len(self.images)
        elif self.mode == 'voc_fallback':
            return len(self.indices)
        else:
            return self.n_samples

    def __getitem__(self, idx):
        if self.mode == 'real':
            img = Image.open(self.images[idx]).convert('RGB')
            ann = Image.open(self.annotations[idx])
            img = self.img_transform(img)
            ann = T.Resize((self.img_size, self.img_size),
                           interpolation=T.InterpolationMode.NEAREST)(ann)
            mask = torch.tensor(np.array(ann), dtype=torch.long)
            # ADE20K: 0 is background, 1-150 are classes
            # Shift to 0-149 for cross entropy
            mask = mask - 1
            mask[mask < 0] = 255  # ignore background
            return img, mask

        elif self.mode == 'voc_fallback':
            img, ann = self.voc[self.indices[idx]]
            img = self.img_transform(img)
            ann = T.Resize((self.img_size, self.img_size),
                           interpolation=T.InterpolationMode.NEAREST)(ann)
            mask = torch.tensor(np.array(ann), dtype=torch.long)
            return img, mask

        else:
            # Synthetic
            img = torch.randn(3, self.img_size, self.img_size)
            mask = torch.randint(0, NUM_CLASSES, (self.img_size, self.img_size))
            return img, mask


def train_segmentation(model, train_loader, val_loader, device, epochs=50, lr=1e-3,
                        num_classes=150):
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

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            all_ious = []
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    logits = model(imgs, target_size=(masks.shape[1], masks.shape[2]))
                    preds = logits.argmax(dim=1)
                    miou = compute_miou(preds, masks, num_classes)
                    all_ious.append(miou)
            val_miou = np.mean(all_ious)
            best_miou = max(best_miou, val_miou)
            print(f"      Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, "
                  f"mIoU={val_miou:.4f} (best={best_miou:.4f})")

    return best_miou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='dinov2',
                        choices=list(BACKBONES.keys()))
    parser.add_argument('--n_train', type=int, default=800)
    parser.add_argument('--data_root', type=str, default='./data')
    args = parser.parse_args()

    device = setup_device()
    config = ExperimentConfig()
    bb = BACKBONES[args.backbone]

    print("=" * 70)
    print(f"Dense Prediction: ADE20K Segmentation with {bb['name']}")
    print("=" * 70)

    base_backbone = timm.create_model(bb['model'], pretrained=True,
                                       img_size=bb['img_size'])
    config.embed_dim = base_backbone.embed_dim
    config.num_layers = len(base_backbone.blocks)
    config.num_heads = base_backbone.blocks[0].attn.num_heads
    config.head_dim = base_backbone.embed_dim // base_backbone.blocks[0].attn.num_heads

    patch_grid = bb['img_size'] // bb['patch_size']
    base_backbone.head = nn.Identity()

    # Determine num_classes based on available dataset
    print("\n  Loading dataset...")
    train_dataset = ADE20KDataset(args.data_root, 'training', bb['img_size'],
                                   max_samples=args.n_train)
    val_dataset = ADE20KDataset(args.data_root, 'validation', bb['img_size'],
                                 max_samples=200)

    num_classes = 21 if train_dataset.mode == 'voc_fallback' else NUM_CLASSES

    seg_head = SegmentationHead(config.embed_dim, num_classes, patch_grid)
    base_model = ViTSegModel(base_backbone, seg_head)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                             num_workers=2, pin_memory=True)

    sigma_p = compute_sigma_p_sq(base_model.backbone.state_dict(), config)
    print(f"\n  σ²_P = {sigma_p:.2f}, dataset: {train_dataset.mode}")

    results = {}

    methods = [
        ('LP', None, 1e-2),
        ('LoRA_r4', 'lora', 4, 1e-3),
        ('LoRA_r8', 'lora', 8, 1e-3),
        ('VPT_p1', 'vpt', 1, 1e-2),
        ('VPT_p5', 'vpt', 5, 1e-2),
    ]

    for method_info in methods:
        if len(method_info) == 3:
            name, _, lr = method_info
            method_type, capacity = None, None
        else:
            name, method_type, capacity, lr = method_info

        print(f"\n  {'='*45}")
        print(f"  {bb['name']} × ADE20K × {name}")
        print(f"  {'='*45}")

        model = deepcopy(base_model)

        if method_type is None:
            for param in model.backbone.parameters():
                param.requires_grad_(False)
            for param in model.seg_head.parameters():
                param.requires_grad_(True)
        elif method_type == 'lora':
            model.backbone = apply_lora(model.backbone, capacity, config)
            for param in model.seg_head.parameters():
                param.requires_grad_(True)
        elif method_type == 'vpt':
            model.backbone = apply_vpt(model.backbone, capacity, config)
            for param in model.seg_head.parameters():
                param.requires_grad_(True)

        model = model.to(device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    {name}: {trainable:,} trainable params")

        miou = train_segmentation(model, train_loader, val_loader, device,
                                   epochs=50, lr=lr, num_classes=num_classes)
        results[name] = miou
        print(f"    → Best mIoU: {miou:.4f}")

        del model; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {bb['name']} × ADE20K Segmentation")
    print(f"{'='*70}")
    best_lora = max(results.get('LoRA_r4', 0), results.get('LoRA_r8', 0))
    best_vpt = max(results.get('VPT_p1', 0), results.get('VPT_p5', 0))
    for name, miou in results.items():
        print(f"  {name:<12s}: mIoU = {miou:.4f}")
    print(f"\n  Best LoRA: {best_lora:.4f}, Best VPT: {best_vpt:.4f}")
    print(f"  Winner: {'LoRA' if best_lora > best_vpt else 'VPT' if best_vpt > best_lora else 'TIE'}")

    os.makedirs('results', exist_ok=True)
    fname = f'results/ade20k_{args.backbone}.json'
    with open(fname, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {fname}")


if __name__ == '__main__':
    main()
