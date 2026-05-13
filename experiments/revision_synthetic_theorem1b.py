"""
Synthetic Validation of Theorem 1(b): VPT Advantage on Attention Steering

Creates a controlled task matching the theorem's construction:
- 2 classes that share the same features but swap their positions
- Tests whether VPT can separate while W_V-only LoRA cannot

Uses a real ViT backbone but constructs the input to match the theorem.

Protocol:
1. Take random image pairs (img_A, img_B) from a dataset
2. Class 1: left=img_A, right=img_B (concatenated as 2-patch input)
3. Class 2: left=img_B, right=img_A
4. Both classes contain the same visual content, just swapped
5. Compare LoRA(W_V only), LoRA(all), VPT

Expected results matching theorem:
- LoRA_WV: ~50% (chance), cannot use position
- LoRA_all: >50%, can partially exploit position via W_K
- VPT: >50%, steers attention based on position-feature combo

Usage:
    python revision_synthetic_theorem1b.py
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
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

from config import ExperimentConfig, setup_device


class SwappedFeatureDataset(Dataset):
    """Constructs the Theorem 1(b) task from real images.
    
    Takes pairs of images and creates:
    Class 0: [img_A, img_B] as a 2-patch sequence
    Class 1: [img_B, img_A] as a 2-patch sequence (swapped)
    
    Both classes contain the same pair of features, just at different positions.
    """
    def __init__(self, base_dataset, n_pairs=500, img_size=224, seed=42):
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(base_dataset))[:2 * n_pairs]
        
        self.pairs = []
        for i in range(n_pairs):
            idx_a, idx_b = indices[2*i], indices[2*i+1]
            img_a = base_dataset[idx_a][0]
            img_b = base_dataset[idx_b][0]
            
            if not isinstance(img_a, torch.Tensor):
                img_a = self.transform(img_a)
                img_b = self.transform(img_b)
            
            # Class 0: A left, B right
            # Class 1: B left, A right
            self.pairs.append((img_a, img_b))
    
    def __len__(self):
        return len(self.pairs) * 2  # Each pair generates 2 samples
    
    def __getitem__(self, idx):
        pair_idx = idx // 2
        is_swapped = idx % 2  # 0 = original order, 1 = swapped
        
        img_a, img_b = self.pairs[pair_idx]
        
        if is_swapped:
            # Class 1: swap left/right
            combined = torch.cat([img_b, img_a], dim=2)  # concat along width
        else:
            # Class 0: original order
            combined = torch.cat([img_a, img_b], dim=2)  # concat along width
        
        # Resize to model input size (the ViT will patchify this)
        combined = F.interpolate(combined.unsqueeze(0), 
                                  size=(self.img_size, self.img_size),
                                  mode='bilinear', align_corners=False).squeeze(0)
        
        return combined, is_swapped


def apply_lora_wv_only(model, rank):
    """Apply LoRA to W_V only (as in the theorem)."""
    from exp2_comparison import LoRALayer
    
    for param in model.parameters():
        param.requires_grad_(False)
    
    for block in model.blocks:
        # Only modify the value projection part of qkv
        # timm combines Q,K,V into one qkv matrix
        # We add LoRA to the full qkv but mask to only affect V
        old_qkv = block.attn.qkv
        d = model.embed_dim
        
        # Create a LoRA that only modifies the V portion
        class VOnlyLoRA(nn.Module):
            def __init__(self, original, rank, d):
                super().__init__()
                self.original = original
                d_out = original.weight.shape[0]  # 3*d
                d_in = original.weight.shape[1]   # d
                # Only modify last d rows (V part of QKV)
                self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
                self.lora_B = nn.Parameter(torch.zeros(d, rank))  # d, not 3d
                self.d = d
            
            def forward(self, x):
                out = self.original(x)
                # Add LoRA only to V portion (last d dims of output)
                lora_out = x @ self.lora_A.T @ self.lora_B.T
                out[..., 2*self.d:] = out[..., 2*self.d:] + lora_out
                return out
        
        block.attn.qkv = VOnlyLoRA(old_qkv, rank, d)
    
    for name, param in model.named_parameters():
        if 'lora_' in name or 'head' in name:
            param.requires_grad_(True)
    
    return model


def apply_lora_all(model, rank):
    """Apply LoRA to all attention weights (Q, K, V, proj)."""
    from exp2_comparison import apply_lora
    config = ExperimentConfig()
    config.embed_dim = model.embed_dim
    config.num_layers = len(model.blocks)
    config.num_heads = model.blocks[0].attn.num_heads
    config.head_dim = model.embed_dim // model.blocks[0].attn.num_heads
    return apply_lora(model, rank, config)


def apply_vpt(model, num_prompts):
    """Apply VPT."""
    from exp2_comparison import apply_vpt as _apply_vpt
    config = ExperimentConfig()
    config.embed_dim = model.embed_dim
    config.num_layers = len(model.blocks)
    config.num_heads = model.blocks[0].attn.num_heads
    config.head_dim = model.embed_dim // model.blocks[0].attn.num_heads
    return _apply_vpt(model, num_prompts, config)


def train_and_eval(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    """Train binary classifier and return accuracy."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x).argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += len(y)
            acc = correct / total
            best_acc = max(best_acc, acc)
    
    return best_acc


def main():
    device = setup_device()
    
    print("=" * 70)
    print("Synthetic Validation of Theorem 1(b)")
    print("Task: classify swapped-feature pairs (same content, different order)")
    print("=" * 70)
    
    # Use CIFAR-10 as source of diverse image patches
    base_ds = datasets.CIFAR10(root='./data', train=True, download=True)
    
    backbone_name = 'deit3_base_patch16_224.fb_in1k'
    
    print(f"\n  Loading backbone: {backbone_name}")
    base_model = timm.create_model(backbone_name, pretrained=True, img_size=224).to(device)
    
    # Create swapped-feature dataset
    print("  Creating swapped-feature dataset...")
    train_ds = SwappedFeatureDataset(base_ds, n_pairs=400, img_size=224, seed=42)
    val_ds = SwappedFeatureDataset(base_ds, n_pairs=100, img_size=224, seed=123)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    embed_dim = base_model.embed_dim
    results = {}
    
    methods = [
        ('LP (baseline)', None, 1e-2),
        ('LoRA_WV_r4', lambda m: apply_lora_wv_only(m, 4), 1e-3),
        ('LoRA_WV_r8', lambda m: apply_lora_wv_only(m, 8), 1e-3),
        ('LoRA_all_r4', lambda m: apply_lora_all(m, 4), 1e-3),
        ('LoRA_all_r8', lambda m: apply_lora_all(m, 8), 1e-3),
        ('VPT_p1', lambda m: apply_vpt(m, 1), 1e-2),
        ('VPT_p5', lambda m: apply_vpt(m, 5), 1e-2),
        ('VPT_p10', lambda m: apply_vpt(m, 10), 1e-2),
    ]
    
    for name, apply_fn, lr in methods:
        print(f"\n  {name}:")
        
        model = deepcopy(base_model)
        model.head = nn.Linear(embed_dim, 2).to(device)
        
        if apply_fn is not None:
            model = apply_fn(model)
        else:
            for param in model.parameters():
                param.requires_grad_(False)
            for param in model.head.parameters():
                param.requires_grad_(True)
        
        model = model.to(device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Trainable: {trainable:,}")
        
        acc = train_and_eval(model, train_loader, val_loader, device, epochs=50, lr=lr)
        results[name] = acc
        print(f"    Accuracy: {acc:.3f} {'(chance=0.50)' if acc < 0.55 else ''}")
        
        del model; torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Theorem 1(b) Synthetic Validation")
    print(f"{'='*70}")
    print(f"  Task: classify image pairs by spatial order (same content)")
    print(f"  Chance level: 0.500")
    print()
    
    for name, acc in results.items():
        marker = ""
        if 'WV' in name and acc < 0.55:
            marker = " ← VALIDATES theorem (W_V-only fails)"
        elif 'VPT' in name and acc > 0.55:
            marker = " ← VALIDATES theorem (VPT succeeds)"
        elif 'all' in name and acc > 0.55:
            marker = " ← Joint LoRA partially exploits position"
        print(f"  {name:<20s}: {acc:.3f}{marker}")
    
    print(f"\n  Theory prediction:")
    print(f"    LoRA(W_V only) ≈ 0.50 (cannot exploit position)")
    print(f"    LoRA(all) > 0.50 (may exploit position via W_K)")
    print(f"    VPT > 0.50 (attention steering exploits position)")
    
    os.makedirs('results', exist_ok=True)
    with open('results/synthetic_theorem1b.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to results/synthetic_theorem1b.json")


if __name__ == '__main__':
    main()
