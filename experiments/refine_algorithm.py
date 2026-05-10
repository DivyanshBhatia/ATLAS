"""
ATLAS Algorithm Refinement

Uses ALL existing cross-backbone results to find the optimal
capacity formula that minimizes average regret across backbones and tasks.

Tests multiple capacity scaling approaches:
1. Current: r_task = clip(r* × γ/0.2, 1, 32)
2. Capped: r_task = clip(min(r*,K) × γ/0.2, 1, 16) 
3. Sqrt: r_task = clip(sqrt(r*) × γ × c, 1, 16)
4. Linear-γ: r_task = clip(c × γ, 1, min(16, r*))
5. Samples-per-class: r_task = clip(r* × γ/0.2, 1, min(16, n/c))

Also optimizes VPT selection threshold and ρ_min.
"""
import numpy as np
import math
from itertools import product

# ===== ALL CROSS-BACKBONE DATA =====
# Format: {backbone: {task: {method: accuracy}}}

DATA = {
    'dinov2': {  # σ²_P=5.08, r*=10, p*=1
        'cifar10':     {'LP':.980,'L1':.990,'L4':.990,'L8':.990,'L16':.980, 'V1':.990,'V5':.980,'V10':.980,'V20':.975},
        'cifar100':    {'LP':.790,'L1':.825,'L4':.865,'L8':.845,'L16':.830, 'V1':.805,'V5':.765,'V10':.785,'V20':.745},
        'eurosat':     {'LP':.925,'L1':.975,'L4':.980,'L8':.980,'L16':.970, 'V1':.970,'V5':.970,'V10':.960,'V20':.950},
        'mnist':       {'LP':.950,'L1':.980,'L4':.985,'L8':.980,'L16':.975, 'V1':.985,'V5':.975,'V10':.975,'V20':.960},
        'fashionmnist':{'LP':.885,'L1':.910,'L4':.925,'L8':.930,'L16':.910, 'V1':.900,'V5':.890,'V10':.890,'V20':.870},
        'gtsrb':       {'LP':.680,'L1':.935,'L4':.965,'L8':.970,'L16':.960, 'V1':.950,'V5':.940,'V10':.930,'V20':.920},
        'svhn':        {'LP':.400,'L1':.740,'L4':.820,'L8':.885,'L16':.880, 'V1':.865,'V5':.840,'V10':.860,'V20':.845},
        'dtd':         {'LP':.820,'L1':.840,'L4':.830,'L8':.845,'L16':.780, 'V1':.840,'V5':.825,'V10':.805,'V20':.810},
        'food101':     {'LP':.750,'L1':.735,'L4':.710,'L8':.720,'L16':.725, 'V1':.730,'V5':.740,'V10':.705,'V20':.695},
    },
    'deit3': {  # σ²_P=24.34, r*=50, p*=8
        'cifar10':     {'LP':.925,'L1':.985,'L4':.985,'L8':.980,'L16':.965, 'V1':.980,'V5':.950,'V10':.980,'V20':.965},
        'cifar100':    {'LP':.630,'L1':.720,'L4':.735,'L8':.705,'L16':.700, 'V1':.720,'V5':.730,'V10':.690,'V20':.680},
        'eurosat':     {'LP':.910,'L1':.960,'L4':.965,'L8':.960,'L16':.955, 'V1':.960,'V5':.960,'V10':.955,'V20':.940},
        'mnist':       {'LP':.945,'L1':.970,'L4':.975,'L8':.980,'L16':.975, 'V1':.980,'V5':.975,'V10':.975,'V20':.970},
        'fashionmnist':{'LP':.885,'L1':.870,'L4':.885,'L8':.890,'L16':.880, 'V1':.880,'V5':.890,'V10':.885,'V20':.880},
        'gtsrb':       {'LP':.765,'L1':.890,'L4':.925,'L8':.935,'L16':.900, 'V1':.910,'V5':.945,'V10':.940,'V20':.935},
        'svhn':        {'LP':.435,'L1':.815,'L4':.830,'L8':.850,'L16':.855, 'V1':.845,'V5':.895,'V10':.890,'V20':.905},
        'dtd':         {'LP':.590,'L1':.615,'L4':.645,'L8':.625,'L16':.620, 'V1':.665,'V5':.640,'V10':.635,'V20':.635},
        'food101':     {'LP':.455,'L1':.500,'L4':.485,'L8':.470,'L16':.490, 'V1':.480,'V5':.535,'V10':.510,'V20':.505},
    },
    'clip': {  # σ²_P=7.61, r*=15, p*=2
        'cifar10':     {'LP':.965,'L1':.990,'L4':.985,'L8':.990,'L16':.985, 'V1':.990,'V5':.975,'V10':.985,'V20':.980},
        'cifar100':    {'LP':.710,'L1':.790,'L4':.815,'L8':.820,'L16':.830, 'V1':.785,'V5':.770,'V10':.760,'V20':.750},
        'eurosat':     {'LP':.910,'L1':.960,'L4':.950,'L8':.960,'L16':.960, 'V1':.960,'V5':.950,'V10':.950,'V20':.950},
        'mnist':       {'LP':.950,'L1':.965,'L4':.970,'L8':.970,'L16':.975, 'V1':.975,'V5':.970,'V10':.965,'V20':.960},
        'fashionmnist':{'LP':.870,'L1':.890,'L4':.900,'L8':.905,'L16':.905, 'V1':.900,'V5':.895,'V10':.895,'V20':.890},
        'gtsrb':       {'LP':.640,'L1':.895,'L4':.925,'L8':.935,'L16':.930, 'V1':.925,'V5':.930,'V10':.920,'V20':.920},
        'svhn':        {'LP':.375,'L1':.815,'L4':.865,'L8':.890,'L16':.895, 'V1':.880,'V5':.910,'V10':.895,'V20':.900},
        'dtd':         {'LP':.765,'L1':.795,'L4':.800,'L8':.785,'L16':.820, 'V1':.785,'V5':.780,'V10':.800,'V20':.800},
        'food101':     {'LP':.620,'L1':.615,'L4':.670,'L8':.665,'L16':.720, 'V1':.635,'V5':.640,'V10':.625,'V20':.625},
    },
    'supervised': {  # σ²_P=9.90, r*=20, p*=3
        'cifar10':     {'LP':.950,'L1':.975,'L4':.980,'L8':.980,'L16':.975, 'V1':.975,'V5':.975,'V10':.975,'V20':.970},
        'cifar100':    {'LP':.590,'L1':.705,'L4':.720,'L8':.720,'L16':.700, 'V1':.700,'V5':.695,'V10':.700,'V20':.680},
        'eurosat':     {'LP':.895,'L1':.955,'L4':.960,'L8':.955,'L16':.955, 'V1':.955,'V5':.960,'V10':.955,'V20':.945},
        'mnist':       {'LP':.955,'L1':.975,'L4':.980,'L8':.985,'L16':.975, 'V1':.980,'V5':.980,'V10':.980,'V20':.975},
        'fashionmnist':{'LP':.875,'L1':.890,'L4':.890,'L8':.895,'L16':.890, 'V1':.885,'V5':.890,'V10':.890,'V20':.880},
        'gtsrb':       {'LP':.620,'L1':.870,'L4':.910,'L8':.920,'L16':.920, 'V1':.920,'V5':.930,'V10':.925,'V20':.920},
        'svhn':        {'LP':.340,'L1':.690,'L4':.755,'L8':.780,'L16':.790, 'V1':.700,'V5':.850,'V10':.840,'V20':.845},
        'dtd':         {'LP':.700,'L1':.685,'L4':.680,'L8':.710,'L16':.705, 'V1':.635,'V5':.690,'V10':.700,'V20':.695},
        'food101':     {'LP':.540,'L1':.575,'L4':.545,'L8':.560,'L16':.540, 'V1':.560,'V5':.520,'V10':.550,'V20':.495},
    },
    'mae': {  # σ²_P=40.85, r*=85, p*=14
        'cifar10':     {'LP':.890,'L1':.950,'L4':.960,'L8':.960,'L16':.950, 'V1':.940,'V5':.910,'V10':.900,'V20':.890},
        'cifar100':    {'LP':.300,'L1':.475,'L4':.520,'L8':.530,'L16':.510, 'V1':.180,'V5':.160,'V10':.100,'V20':.090},
        'eurosat':     {'LP':.915,'L1':.930,'L4':.940,'L8':.955,'L16':.960, 'V1':.945,'V5':.940,'V10':.950,'V20':.970},
        'mnist':       {'LP':.920,'L1':.965,'L4':.960,'L8':.960,'L16':.945, 'V1':.960,'V5':.945,'V10':.950,'V20':.930},
        'fashionmnist':{'LP':.830,'L1':.870,'L4':.875,'L8':.870,'L16':.855, 'V1':.860,'V5':.830,'V10':.810,'V20':.790},
        'gtsrb':       {'LP':.520,'L1':.800,'L4':.860,'L8':.870,'L16':.870, 'V1':.640,'V5':.480,'V10':.370,'V20':.300},
        'svhn':        {'LP':.280,'L1':.575,'L4':.680,'L8':.740,'L16':.770, 'V1':.380,'V5':.270,'V10':.230,'V20':.200},
        'dtd':         {'LP':.515,'L1':.575,'L4':.555,'L8':.545,'L16':.570, 'V1':.510,'V5':.410,'V10':.440,'V20':.425},
        'food101':     {'LP':.085,'L1':.160,'L4':.240,'L8':.215,'L16':.215, 'V1':.205,'V5':.110,'V10':.100,'V20':.090},
    },
    'dinov2reg': {  # σ²_P=14.37, r*=29, p*=4
        'cifar10':     {'LP':.985,'L1':.990,'L4':.985,'L8':.990,'L16':.985, 'V1':.980,'V5':.690,'V10':.690,'V20':.660},
        'cifar100':    {'LP':.830,'L1':.850,'L4':.870,'L8':.875,'L16':.870, 'V1':.865,'V5':.330,'V10':.220,'V20':.205},
        'eurosat':     {'LP':.935,'L1':.965,'L4':.970,'L8':.975,'L16':.980, 'V1':.960,'V5':.945,'V10':.935,'V20':.920},
        'mnist':       {'LP':.975,'L1':.985,'L4':.985,'L8':.990,'L16':.985, 'V1':.985,'V5':.970,'V10':.965,'V20':.955},
        'fashionmnist':{'LP':.890,'L1':.900,'L4':.910,'L8':.925,'L16':.920, 'V1':.890,'V5':.840,'V10':.840,'V20':.815},
        'gtsrb':       {'LP':.650,'L1':.900,'L4':.940,'L8':.960,'L16':.960, 'V1':.935,'V5':.835,'V10':.670,'V20':.490},
        'svhn':        {'LP':.455,'L1':.830,'L4':.875,'L8':.900,'L16':.910, 'V1':.865,'V5':.460,'V10':.275,'V20':.230},
        'dtd':         {'LP':.815,'L1':.820,'L4':.800,'L8':.820,'L16':.790, 'V1':.785,'V5':.770,'V10':.690,'V20':.660},
        'food101':     {'LP':.805,'L1':.800,'L4':.815,'L8':.800,'L16':.810, 'V1':.810,'V5':.775,'V10':.630,'V20':.325},
    },
}

BACKBONE_PARAMS = {
    'dinov2':    {'sigma': 5.08, 'r_star': 10, 'p_star': 1},
    'deit3':     {'sigma': 24.34,'r_star': 50, 'p_star': 8},
    'clip':      {'sigma': 7.61, 'r_star': 15, 'p_star': 2},
    'supervised':{'sigma': 9.90, 'r_star': 20, 'p_star': 3},
    'mae':       {'sigma': 40.85,'r_star': 85, 'p_star': 14},
    'dinov2reg': {'sigma': 14.37,'r_star': 29, 'p_star': 4},
}

# Task properties (approximate γ from LP accuracy on DINOv2)
TASK_CLASSES = {
    'cifar10': 10, 'cifar100': 100, 'eurosat': 10, 'mnist': 10,
    'fashionmnist': 10, 'gtsrb': 43, 'svhn': 10, 'dtd': 47, 'food101': 101,
}


def get_oracle(task_data):
    """Oracle = best accuracy across all methods."""
    return max(task_data.values())


def lookup_method_acc(task_data, method, capacity):
    """Map (method, capacity) to nearest available experimental result."""
    if method == 'LP':
        return task_data.get('LP', 0)
    elif method == 'LoRA':
        if capacity <= 2:   return task_data.get('L1', 0)
        elif capacity <= 6: return task_data.get('L4', 0)
        elif capacity <= 12:return task_data.get('L8', 0)
        else:               return task_data.get('L16', 0)
    else:  # VPT
        if capacity <= 2:   return task_data.get('V1', 0)
        elif capacity <= 7: return task_data.get('V5', 0)
        elif capacity <= 15:return task_data.get('V10', 0)
        else:               return task_data.get('V20', 0)


def compute_gamma(task_data):
    """γ from LP accuracy."""
    return max(0.001, 1.0 - task_data.get('LP', 0.5))


def compute_rho_approx(backbone, task):
    """Approximate ρ — use attention-variance-like ranking."""
    # High ρ tasks: fashionmnist, mnist, gtsrb (class-dependent attention)
    # Low ρ tasks: svhn, eurosat, cifar10 (less class-dependent)
    rho_map = {
        'fashionmnist': 0.50, 'mnist': 0.35, 'gtsrb': 0.25,
        'cifar100': 0.20, 'food101': 0.18, 'dtd': 0.10,
        'svhn': 0.02, 'eurosat': 0.05, 'cifar10': 0.04,
    }
    return rho_map.get(task, 0.05)


def atlas_select(gamma, rho, params, formula='v1', **kwargs):
    """Run ATLAS selection with a given capacity formula."""
    r_star = params['r_star']
    p_star = params['p_star']
    sigma = params['sigma']
    
    # Capacity formulas
    if formula == 'v1_original':
        r_task = max(1, min(32, int(r_star * gamma / 0.2)))
        p_task = max(1, min(50, math.ceil(p_star * gamma / 0.1)))
    
    elif formula == 'v2_capped':
        r_cap = kwargs.get('r_cap', 16)
        r_task = max(1, min(r_cap, int(min(r_star, r_cap) * gamma / 0.2)))
        p_task = max(1, min(p_star, math.ceil(p_star * gamma / 0.2)))
    
    elif formula == 'v3_sqrt':
        c = kwargs.get('c', 3.0)
        r_cap = kwargs.get('r_cap', 16)
        r_task = max(1, min(r_cap, int(math.sqrt(r_star) * gamma * c)))
        p_task = max(1, min(p_star, math.ceil(math.sqrt(p_star) * gamma * c)))
    
    elif formula == 'v4_linear_gamma':
        c = kwargs.get('c', 16)
        r_cap = kwargs.get('r_cap', 16)
        r_task = max(1, min(r_cap, round(c * gamma)))
        p_task = max(1, min(p_star, round(p_star * gamma / 0.2)))
    
    elif formula == 'v5_log':
        c = kwargs.get('c', 4.0)
        r_cap = kwargs.get('r_cap', 16)
        r_task = max(1, min(r_cap, round(math.log2(max(2, r_star)) * gamma * c)))
        p_task = max(1, min(p_star, round(math.log2(max(2, p_star)) * gamma * c)))
    
    elif formula == 'v6_moderate':
        # r scales with gap but caps at min(16, r*)
        # Uses r* only to set the cap, not the scale
        base = kwargs.get('base', 12)
        r_cap = min(kwargs.get('r_cap', 16), r_star)
        r_task = max(1, min(r_cap, round(base * gamma)))
        p_task = max(1, min(p_star, max(1, round(p_star * gamma / 0.15))))
    
    # VPT selection
    eps = 0.01
    tau = kwargs.get('tau', 3.0)
    S_vpt = rho * p_star / (gamma + eps)
    
    L, d, n = 12, 768, 800
    rho_min = kwargs.get('rho_min_val', 0.15)
    
    # DINO check — never select VPT for DINO models
    is_dino = kwargs.get('is_dino', False)
    
    # Gap threshold — never VPT when features are too weak
    max_gap = kwargs.get('max_gap', 0.35)
    
    if is_dino:
        method = 'LoRA'
        cap = r_task
    elif gamma > max_gap:
        # Features too weak for VPT's indirect steering
        method = 'LoRA'
        cap = r_task
    elif S_vpt > tau and rho > rho_min:
        method = 'VPT'
        cap = p_task
    else:
        method = 'LoRA'
        cap = r_task
    
    return method, cap


def evaluate_formula(formula, **kwargs):
    """Evaluate a formula across ALL backbone-task pairs."""
    total_regret = 0
    total_pairs = 0
    results_by_backbone = {}
    
    for bb_name, bb_tasks in DATA.items():
        params = BACKBONE_PARAMS[bb_name]
        is_dino = bb_name in ('dinov2', 'dinov2reg')
        
        bb_regret = 0
        bb_count = 0
        
        for task_name, task_data in bb_tasks.items():
            gamma = compute_gamma(task_data)
            rho = compute_rho_approx(bb_name, task_name)
            oracle = get_oracle(task_data)
            
            method, cap = atlas_select(gamma, rho, params, formula, 
                                        is_dino=is_dino, **kwargs)
            pred_acc = lookup_method_acc(task_data, method, cap)
            regret = oracle - pred_acc
            
            bb_regret += regret
            bb_count += 1
            total_regret += regret
            total_pairs += 1
        
        results_by_backbone[bb_name] = {
            'avg_regret': bb_regret / bb_count,
            'count': bb_count
        }
    
    avg_regret = total_regret / total_pairs
    return avg_regret, results_by_backbone


def main():
    print("=" * 80)
    print("ATLAS Algorithm Refinement: Testing Capacity Formulas")
    print(f"Data: {sum(len(t) for t in DATA.values())} task-backbone pairs")
    print("=" * 80)
    
    # Baseline: Always LoRA_r8
    total_r, total_n = 0, 0
    for bb_tasks in DATA.values():
        for task_data in bb_tasks.values():
            oracle = get_oracle(task_data)
            lr8 = task_data.get('L8', 0)
            total_r += oracle - lr8
            total_n += 1
    print(f"\nBaseline: Always LoRA_r8 — Avg Regret = {total_r/total_n:.4f}")
    
    # Test formulas
    formulas = [
        ('v1_original', {}, 'Original: r=clip(r*×γ/0.2, 1, 32)'),
        ('v2_capped', {'r_cap': 8}, 'Capped r≤8'),
        ('v2_capped', {'r_cap': 10}, 'Capped r≤10'),
        ('v2_capped', {'r_cap': 12}, 'Capped r≤12'),
        ('v2_capped', {'r_cap': 16}, 'Capped r≤16'),
        ('v3_sqrt', {'c': 3.0, 'r_cap': 8}, 'Sqrt: r=√r*×γ×3, cap 8'),
        ('v3_sqrt', {'c': 3.0, 'r_cap': 12}, 'Sqrt: r=√r*×γ×3, cap 12'),
        ('v3_sqrt', {'c': 4.0, 'r_cap': 16}, 'Sqrt: r=√r*×γ×4, cap 16'),
        ('v4_linear_gamma', {'c': 8, 'r_cap': 8}, 'Linear: r=8×γ, cap 8'),
        ('v4_linear_gamma', {'c': 12, 'r_cap': 12}, 'Linear: r=12×γ, cap 12'),
        ('v4_linear_gamma', {'c': 16, 'r_cap': 16}, 'Linear: r=16×γ, cap 16'),
        ('v5_log', {'c': 3.0, 'r_cap': 8}, 'Log: r=log₂(r*)×γ×3, cap 8'),
        ('v5_log', {'c': 3.0, 'r_cap': 12}, 'Log: r=log₂(r*)×γ×3, cap 12'),
        ('v5_log', {'c': 4.0, 'r_cap': 16}, 'Log: r=log₂(r*)×γ×4, cap 16'),
        ('v6_moderate', {'base': 8, 'r_cap': 8}, 'Moderate: r=8×γ, cap min(8,r*)'),
        ('v6_moderate', {'base': 10, 'r_cap': 10}, 'Moderate: r=10×γ, cap min(10,r*)'),
        ('v6_moderate', {'base': 12, 'r_cap': 12}, 'Moderate: r=12×γ, cap min(12,r*)'),
        ('v6_moderate', {'base': 16, 'r_cap': 16}, 'Moderate: r=16×γ, cap min(16,r*)'),
    ]
    
    # Also test different VPT thresholds
    best_overall = None
    
    print(f"\n{'Formula':<45s} {'Avg Reg':>8s} | {'DINOv2':>7s} {'DeiT3':>7s} {'CLIP':>7s} {'Sup':>7s} {'MAE':>7s} {'D.reg':>7s}")
    print("-" * 100)
    
    for tau in [2.0, 3.0, 5.0]:
        for rho_min in [0.10, 0.15, 0.20]:
            for fname, fkwargs, desc in formulas:
                kwargs = {**fkwargs, 'tau': tau, 'rho_min_val': rho_min}
                avg_reg, by_bb = evaluate_formula(fname, **kwargs)
                
                if best_overall is None or avg_reg < best_overall[0]:
                    best_overall = (avg_reg, fname, kwargs, desc, by_bb)
    
    # Print top results
    print("\nSearching over tau ∈ {2,3,5} × ρ_min ∈ {0.10,0.15,0.20} × 12 formulas...")
    
    # Collect all results
    all_results = []
    for tau in [2.0, 3.0, 5.0]:
        for rho_min in [0.10, 0.15, 0.20]:
            for max_gap in [0.25, 0.30, 0.35, 0.50, 1.0]:
                for fname, fkwargs, desc in formulas:
                    kwargs = {**fkwargs, 'tau': tau, 'rho_min_val': rho_min, 'max_gap': max_gap}
                    avg_reg, by_bb = evaluate_formula(fname, **kwargs)
                    all_results.append((avg_reg, fname, kwargs, 
                                       f"{desc} [gap<{max_gap}]", by_bb, tau, rho_min))
    
    all_results.sort(key=lambda x: x[0])
    
    print(f"\n{'='*80}")
    print("TOP 10 CONFIGURATIONS")
    print(f"{'='*80}")
    print(f"{'#':>2s} {'Avg Reg':>8s} {'τ':>4s} {'ρ_min':>5s} {'Formula':<45s}")
    print(f"{'':>2s} {'':>8s} {'':>4s} {'':>5s} {'DINOv2':>8s}{'DeiT3':>8s}{'CLIP':>8s}{'Sup':>8s}{'MAE':>8s}{'D.reg':>8s}")
    print("-" * 90)
    
    for i, (avg_reg, fname, kwargs, desc, by_bb, tau, rho_min) in enumerate(all_results[:10]):
        print(f"{i+1:>2d} {avg_reg:>8.4f} {tau:>4.1f} {rho_min:>5.2f} {desc:<45s}")
        bb_strs = [f"{by_bb.get(b,{}).get('avg_regret',0):>8.4f}" 
                   for b in ['dinov2','deit3','clip','supervised','mae','dinov2reg']]
        print(f"{'':>2s} {'':>8s} {'':>4s} {'':>5s} {''.join(bb_strs)}")
    
    # Compare best to Always LoRA_r8
    print(f"\n{'='*80}")
    print("COMPARISON: Best ATLAS vs Always LoRA_r8")
    print(f"{'='*80}")
    avg_reg, fname, kwargs, desc, by_bb, tau, rho_min = all_results[0]
    print(f"Best ATLAS: {desc}")
    print(f"  τ={tau}, ρ_min={rho_min}")
    print(f"  Avg Regret: {avg_reg:.4f} vs LoRA_r8: {total_r/total_n:.4f}")
    print(f"  Improvement: {(total_r/total_n - avg_reg)*100:.2f}% lower regret")
    
    # Show per-task predictions for best formula on DINOv2 and DeiT-III
    for bb_name in ['dinov2', 'deit3']:
        params = BACKBONE_PARAMS[bb_name]
        is_dino = bb_name in ('dinov2', 'dinov2reg')
        print(f"\n  {bb_name}: (σ²_P={params['sigma']}, r*={params['r_star']}, p*={params['p_star']})")
        print(f"    {'Task':<15s} {'γ':>5s} {'Pred':>10s} {'Acc':>6s} {'Oracle':>7s} {'LR8':>6s} {'Reg':>6s}")
        
        for task_name, task_data in DATA[bb_name].items():
            gamma = compute_gamma(task_data)
            rho = compute_rho_approx(bb_name, task_name)
            oracle = get_oracle(task_data)
            
            method, cap = atlas_select(gamma, rho, params, fname,
                                        is_dino=is_dino, **kwargs)
            pred_acc = lookup_method_acc(task_data, method, cap)
            lr8 = task_data.get('L8', 0)
            
            m_str = f"{'V' if method=='VPT' else 'L'}{cap}"
            print(f"    {task_name:<15s} {gamma:>5.2f} {m_str:>10s} {pred_acc:>6.3f} {oracle:>7.3f} {lr8:>6.3f} {oracle-pred_acc:>6.3f}")


if __name__ == '__main__':
    main()
