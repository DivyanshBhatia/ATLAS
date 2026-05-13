"""
Synthetic Validation of Theorem 1(b) — Improved Design

Previous attempt failed because concatenated images create 196 blended patches.
This version directly injects controlled tokens into the ViT, matching the
theorem's 2-token construction EXACTLY:

  Class 1: [CLS, a+pos_1, b+pos_2]
  Class 2: [CLS, b+pos_1, a+pos_2]

where a, b are real feature vectors extracted from the pretrained model.
The ViT processes only 3 tokens (CLS + 2 patches), making the experiment
a direct empirical test of the theorem.

Usage:
    python revision_synthetic_theorem1b_v2.py
    python revision_synthetic_theorem1b_v2.py --backbone dinov2
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
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from config import ExperimentConfig, setup_device


class TokenSwapDataset(Dataset):
    """Constructs the exact Theorem 1(b) task at the token level.
    
    Extracts pairs of patch-embedding vectors (a, b) from real images,
    then creates:
      Class 0: token sequence [a, b] (with positional embeddings added later)
      Class 1: token sequence [b, a] (swapped positions)
    
    The ViT processes these 2-token sequences directly.
    """
    def __init__(self, feature_pairs, seed=42):
        """
        feature_pairs: list of (a, b) tensor pairs, each of shape (d,)
        """
        self.pairs = feature_pairs
        rng = np.random.RandomState(seed)
        self.order = rng.permutation(len(self.pairs))
    
    def __len__(self):
        return len(self.pairs) * 2
    
    def __getitem__(self, idx):
        pair_idx = idx // 2
        is_swapped = idx % 2
        
        a, b = self.pairs[pair_idx]
        
        if is_swapped:
            tokens = torch.stack([b, a])  # (2, d) — swapped
        else:
            tokens = torch.stack([a, b])  # (2, d) — original
        
        return tokens, is_swapped


class MinimalViTClassifier(nn.Module):
    """Wraps a ViT to process exactly 2 patch tokens.
    
    Bypasses patch embedding — takes pre-extracted features directly.
    Adds CLS token and positional embeddings, runs through blocks.
    """
    def __init__(self, vit_model, num_classes=2):
        super().__init__()
        self.blocks = vit_model.blocks
        self.norm = vit_model.norm
        self.embed_dim = vit_model.embed_dim
        
        # Use the pretrained CLS token
        self.cls_token = vit_model.cls_token  # (1, 1, d)
        
        # Use first 3 positional embeddings (CLS + 2 patches)
        full_pos = vit_model.pos_embed  # (1, 1+N, d)
        self.pos_embed = nn.Parameter(full_pos[:, :3, :].clone())  # (1, 3, d)
        
        self.head = nn.Linear(self.embed_dim, num_classes)
        
        # Freeze everything except what we'll explicitly unfreeze
        for param in self.parameters():
            param.requires_grad_(False)
        for param in self.head.parameters():
            param.requires_grad_(True)
    
    def forward(self, patch_tokens):
        """
        patch_tokens: (B, 2, d) — two patch feature vectors
        """
        B = patch_tokens.shape[0]
        
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d)
        x = torch.cat([cls, patch_tokens], dim=1)  # (B, 3, d)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Run through transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # Classify from CLS token
        cls_out = x[:, 0]
        return self.head(cls_out)


def apply_lora_wv_only(model, rank):
    """LoRA on W_V only — should fail per theorem."""
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.head.parameters():
        param.requires_grad_(True)
    
    d = model.embed_dim
    
    for block in model.blocks:
        old_qkv = block.attn.qkv
        
        class VOnlyLoRA(nn.Module):
            def __init__(self, original, rank, d):
                super().__init__()
                self.original = original
                d_in = original.weight.shape[1]
                self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
                self.lora_B = nn.Parameter(torch.zeros(d, rank))
                self.d = d
            
            def forward(self, x):
                out = self.original(x)
                lora_out = x @ self.lora_A.T @ self.lora_B.T
                out = out.clone()
                out[..., 2*self.d:] = out[..., 2*self.d:] + lora_out
                return out
        
        block.attn.qkv = VOnlyLoRA(old_qkv, rank, d)
    
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad_(True)
    
    return model


def apply_lora_all(model, rank):
    """LoRA on all attention weights — may partially work."""
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.head.parameters():
        param.requires_grad_(True)
    
    for block in model.blocks:
        for attr_name in ['qkv', 'proj']:
            old = getattr(block.attn, attr_name)
            d_out, d_in = old.weight.shape
            
            class LoRALayer(nn.Module):
                def __init__(self, original, rank):
                    super().__init__()
                    self.original = original
                    d_out, d_in = original.weight.shape
                    self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
                    self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
                
                def forward(self, x):
                    return self.original(x) + x @ self.lora_A.T @ self.lora_B.T
            
            setattr(block.attn, attr_name, LoRALayer(old, rank))
    
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad_(True)
    
    return model


def apply_vpt_minimal(model, num_prompts):
    """VPT for minimal 3-token ViT."""
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.head.parameters():
        param.requires_grad_(True)
    
    d = model.embed_dim
    
    new_blocks = nn.Sequential()
    for i, block in enumerate(model.blocks):
        class VPTBlock(nn.Module):
            def __init__(self, block, np, d):
                super().__init__()
                self.block = block
                self.prompts = nn.Parameter(torch.randn(1, np, d) * 0.02)
                self.np = np
            
            def forward(self, x):
                B = x.shape[0]
                p = self.prompts.expand(B, -1, -1)
                x = torch.cat([x[:, :1], p, x[:, 1:]], dim=1)
                x = self.block(x)
                x = torch.cat([x[:, :1], x[:, 1+self.np:]], dim=1)
                return x
        
        new_blocks.add_module(str(i), VPTBlock(block, num_prompts, d))
    model.blocks = new_blocks
    
    for name, param in model.named_parameters():
        if 'prompts' in name:
            param.requires_grad_(True)
    
    return model


def extract_feature_pairs(backbone_name, img_size, n_pairs=500, seed=42):
    """Extract pairs of patch embedding vectors from real images."""
    model = timm.create_model(backbone_name, pretrained=True, img_size=img_size)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    
    all_embeddings = []
    with torch.no_grad():
        for imgs, _ in loader:
            # Get patch embeddings (before transformer blocks)
            x = model.patch_embed(imgs)  # (B, N, d)
            all_embeddings.append(x)
            if len(all_embeddings) * 64 >= 2 * n_pairs + 100:
                break
    
    all_embeddings = torch.cat(all_embeddings, dim=0)  # (total_imgs, N, d)
    
    # For each pair: pick two different patches from different images
    rng = np.random.RandomState(seed)
    pairs = []
    indices = rng.permutation(len(all_embeddings))
    
    for i in range(n_pairs):
        img_a_idx = indices[2*i]
        img_b_idx = indices[2*i + 1]
        
        # Take patch 0 from image A and patch 1 from image B
        # (ensures a ≠ b since they come from different images)
        patch_a = rng.randint(0, all_embeddings.shape[1])
        patch_b = rng.randint(0, all_embeddings.shape[1])
        
        a = all_embeddings[img_a_idx, patch_a].clone()
        b = all_embeddings[img_b_idx, patch_b].clone()
        
        # Ensure they're different enough
        if (a - b).norm() > 0.1:
            pairs.append((a, b))
    
    del model
    torch.cuda.empty_cache()
    
    print(f"  Extracted {len(pairs)} feature pairs (d={all_embeddings.shape[2]})")
    return pairs


def train_and_eval(model, train_loader, val_loader, device, epochs=100, lr=1e-3):
    """Train and evaluate."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            logits = model(tokens)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for tokens, labels in val_loader:
                    tokens, labels = tokens.to(device), labels.to(device)
                    preds = model(tokens).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += len(labels)
            acc = correct / total
            best_acc = max(best_acc, acc)
            print(f"      Epoch {epoch+1}: acc={acc:.3f} (best={best_acc:.3f})")
    
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='deit3',
                        choices=['deit3', 'dinov2', 'clip', 'supervised'])
    args = parser.parse_args()
    
    MODELS = {
        'deit3': ('deit3_base_patch16_224.fb_in1k', 224),
        'dinov2': ('vit_base_patch14_dinov2.lvd142m', 518),
        'clip': ('vit_base_patch16_clip_224.openai', 224),
        'supervised': ('vit_base_patch16_224.augreg_in1k', 224),
    }
    
    device = setup_device()
    backbone_name, img_size = MODELS[args.backbone]
    
    print("=" * 70)
    print(f"Synthetic Theorem 1(b) — Token-Level Construction")
    print(f"Backbone: {backbone_name}")
    print(f"Task: [CLS, a+pos1, b+pos2] vs [CLS, b+pos1, a+pos2]")
    print("=" * 70)
    
    # Extract feature pairs from real images
    print("\n  Extracting feature pairs...")
    pairs = extract_feature_pairs(backbone_name, img_size, n_pairs=500, seed=42)
    
    train_pairs = pairs[:400]
    val_pairs = pairs[400:]
    
    train_ds = TokenSwapDataset(train_pairs, seed=42)
    val_ds = TokenSwapDataset(val_pairs, seed=123)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Load backbone for the minimal ViT
    base_vit = timm.create_model(backbone_name, pretrained=True, img_size=img_size)
    
    results = {}
    
    methods = [
        ('LP (head only)',    None,                                1e-2),
        ('LoRA_WV_r4',       lambda m: apply_lora_wv_only(m, 4),  1e-3),
        ('LoRA_WV_r8',       lambda m: apply_lora_wv_only(m, 8),  1e-3),
        ('LoRA_all_r4',      lambda m: apply_lora_all(m, 4),      1e-3),
        ('LoRA_all_r8',      lambda m: apply_lora_all(m, 8),      1e-3),
        ('VPT_p1',           lambda m: apply_vpt_minimal(m, 1),   1e-2),
        ('VPT_p5',           lambda m: apply_vpt_minimal(m, 5),   1e-2),
        ('VPT_p10',          lambda m: apply_vpt_minimal(m, 10),  1e-2),
    ]
    
    for name, apply_fn, lr in methods:
        print(f"\n  {name}:")
        
        model = MinimalViTClassifier(deepcopy(base_vit), num_classes=2).to(device)
        
        if apply_fn is not None:
            model = apply_fn(model)
        
        model = model.to(device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Trainable: {trainable:,}")
        
        acc = train_and_eval(model, train_loader, val_loader, device, epochs=100, lr=lr)
        results[name] = acc
        
        marker = ""
        if 'WV' in name and acc < 0.55:
            marker = " ← VALIDATES theorem (W_V-only ≈ chance)"
        elif 'VPT' in name and acc > 0.60:
            marker = " ← VALIDATES theorem (VPT separates)"
        elif 'all' in name and acc > 0.55:
            marker = " ← Joint LoRA partially exploits position"
        print(f"    Final: {acc:.3f}{marker}")
        
        del model; torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Theorem 1(b) Token-Level Validation")
    print(f"{'='*70}")
    print(f"  Chance level: 0.500")
    print(f"  Backbone: {args.backbone}\n")
    
    for name, acc in results.items():
        bar = '█' * int(acc * 40)
        print(f"  {name:<20s}: {acc:.3f}  {bar}")
    
    print(f"\n  Theory predictions:")
    print(f"    LoRA(W_V only): ≈ 0.50  (cannot change attention on same-multiset tokens)")
    print(f"    LoRA(all):      > 0.50  (W_K modification may exploit position)")
    print(f"    VPT:            > 0.50  (prompt steering exploits position-feature combos)")
    
    os.makedirs('results', exist_ok=True)
    fname = f'results/synthetic_theorem1b_v2_{args.backbone}.json'
    with open(fname, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {fname}")


if __name__ == '__main__':
    main()
