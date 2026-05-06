"""Generate exp2_single_scale.json from conversation results."""
import json

results = {
    "cifar10": {
        "category": "natural", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.980},
            "LoRA_r1": {"accuracy": 0.980, "rank": 1},
            "LoRA_r2": {"accuracy": 0.985, "rank": 2},
            "LoRA_r4": {"accuracy": 0.980, "rank": 4},
            "LoRA_r8": {"accuracy": 0.985, "rank": 8},
            "LoRA_r16": {"accuracy": 0.990, "rank": 16},
            "LoRA_r32": {"accuracy": 0.990, "rank": 32},
            "VPT_p1": {"accuracy": 0.990, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.980, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.980, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.975, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.950, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.985},
            "Adapter_r32": {"accuracy": 0.985},
            "Adapter_r64": {"accuracy": 0.985}
        }
    },
    "dtd": {
        "category": "natural", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.690},
            "LoRA_r1": {"accuracy": 0.750, "rank": 1},
            "LoRA_r2": {"accuracy": 0.735, "rank": 2},
            "LoRA_r4": {"accuracy": 0.760, "rank": 4},
            "LoRA_r8": {"accuracy": 0.755, "rank": 8},
            "LoRA_r16": {"accuracy": 0.760, "rank": 16},
            "LoRA_r32": {"accuracy": 0.775, "rank": 32},
            "VPT_p1": {"accuracy": 0.745, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.740, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.705, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.730, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.700, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.765},
            "Adapter_r32": {"accuracy": 0.725},
            "Adapter_r64": {"accuracy": 0.780}
        }
    },
    "oxford_flowers102": {
        "category": "natural", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 1.000},
            "LoRA_r1": {"accuracy": 1.000, "rank": 1},
            "LoRA_r2": {"accuracy": 1.000, "rank": 2},
            "LoRA_r4": {"accuracy": 1.000, "rank": 4},
            "LoRA_r8": {"accuracy": 1.000, "rank": 8},
            "LoRA_r16": {"accuracy": 0.995, "rank": 16},
            "LoRA_r32": {"accuracy": 0.995, "rank": 32},
            "VPT_p1": {"accuracy": 1.000, "n_prompts": 1},
            "VPT_p5": {"accuracy": 1.000, "n_prompts": 5},
            "VPT_p10": {"accuracy": 1.000, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.995, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.985, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 1.000},
            "Adapter_r32": {"accuracy": 1.000},
            "Adapter_r64": {"accuracy": 0.995}
        }
    },
    "oxford_iiit_pet": {
        "category": "natural", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.970},
            "LoRA_r1": {"accuracy": 0.965, "rank": 1},
            "LoRA_r2": {"accuracy": 0.970, "rank": 2},
            "LoRA_r4": {"accuracy": 0.955, "rank": 4},
            "LoRA_r8": {"accuracy": 0.955, "rank": 8},
            "LoRA_r16": {"accuracy": 0.955, "rank": 16},
            "LoRA_r32": {"accuracy": 0.940, "rank": 32},
            "VPT_p1": {"accuracy": 0.960, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.955, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.955, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.955, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.860, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.960},
            "Adapter_r32": {"accuracy": 0.965},
            "Adapter_r64": {"accuracy": 0.950}
        }
    },
    "food101": {
        "category": "natural", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.750},
            "LoRA_r1": {"accuracy": 0.730, "rank": 1},
            "LoRA_r2": {"accuracy": 0.745, "rank": 2},
            "LoRA_r4": {"accuracy": 0.735, "rank": 4},
            "LoRA_r8": {"accuracy": 0.715, "rank": 8},
            "LoRA_r16": {"accuracy": 0.735, "rank": 16},
            "LoRA_r32": {"accuracy": 0.690, "rank": 32},
            "VPT_p1": {"accuracy": 0.735, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.730, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.715, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.645, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.115, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.745},
            "Adapter_r32": {"accuracy": 0.715},
            "Adapter_r64": {"accuracy": 0.710}
        }
    },
    "stl10": {
        "category": "natural", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.990},
            "LoRA_r1": {"accuracy": 0.990, "rank": 1},
            "LoRA_r2": {"accuracy": 0.990, "rank": 2},
            "LoRA_r4": {"accuracy": 0.995, "rank": 4},
            "LoRA_r8": {"accuracy": 0.995, "rank": 8},
            "LoRA_r16": {"accuracy": 0.995, "rank": 16},
            "LoRA_r32": {"accuracy": 0.995, "rank": 32},
            "VPT_p1": {"accuracy": 0.990, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.990, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.990, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.990, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.975, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.995},
            "Adapter_r32": {"accuracy": 1.000},
            "Adapter_r64": {"accuracy": 0.995}
        }
    },
    "fgvc_aircraft": {
        "category": "natural", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.490},
            "LoRA_r1": {"accuracy": 0.540, "rank": 1},
            "LoRA_r2": {"accuracy": 0.535, "rank": 2},
            "LoRA_r4": {"accuracy": 0.520, "rank": 4},
            "LoRA_r8": {"accuracy": 0.530, "rank": 8},
            "LoRA_r16": {"accuracy": 0.565, "rank": 16},
            "LoRA_r32": {"accuracy": 0.460, "rank": 32},
            "VPT_p1": {"accuracy": 0.525, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.515, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.480, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.430, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.450, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.490},
            "Adapter_r32": {"accuracy": 0.545},
            "Adapter_r64": {"accuracy": 0.510}
        }
    },
    "eurosat": {
        "category": "specialized", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.925},
            "LoRA_r1": {"accuracy": 0.950, "rank": 1},
            "LoRA_r2": {"accuracy": 0.965, "rank": 2},
            "LoRA_r4": {"accuracy": 0.970, "rank": 4},
            "LoRA_r8": {"accuracy": 0.975, "rank": 8},
            "LoRA_r16": {"accuracy": 0.980, "rank": 16},
            "LoRA_r32": {"accuracy": 0.980, "rank": 32},
            "VPT_p1": {"accuracy": 0.970, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.950, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.970, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.970, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.955, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.980},
            "Adapter_r32": {"accuracy": 0.980},
            "Adapter_r64": {"accuracy": 0.980}
        }
    },
    "pcam": {
        "category": "specialized", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.895},
            "LoRA_r1": {"accuracy": 0.895, "rank": 1},
            "LoRA_r2": {"accuracy": 0.895, "rank": 2},
            "LoRA_r4": {"accuracy": 0.915, "rank": 4},
            "LoRA_r8": {"accuracy": 0.910, "rank": 8},
            "LoRA_r16": {"accuracy": 0.925, "rank": 16},
            "LoRA_r32": {"accuracy": 0.930, "rank": 32},
            "VPT_p1": {"accuracy": 0.895, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.910, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.880, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.915, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.895, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.920},
            "Adapter_r32": {"accuracy": 0.915},
            "Adapter_r64": {"accuracy": 0.910}
        }
    },
    "country211": {
        "category": "specialized", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.070},
            "LoRA_r1": {"accuracy": 0.070, "rank": 1},
            "LoRA_r2": {"accuracy": 0.075, "rank": 2},
            "LoRA_r4": {"accuracy": 0.090, "rank": 4},
            "LoRA_r8": {"accuracy": 0.090, "rank": 8},
            "LoRA_r16": {"accuracy": 0.055, "rank": 16},
            "LoRA_r32": {"accuracy": 0.095, "rank": 32},
            "VPT_p1": {"accuracy": 0.055, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.050, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.075, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.055, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.055, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.080},
            "Adapter_r32": {"accuracy": 0.070},
            "Adapter_r64": {"accuracy": 0.070}
        }
    },
    "svhn": {
        "category": "structured", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.400},
            "LoRA_r1": {"accuracy": 0.770, "rank": 1},
            "LoRA_r2": {"accuracy": 0.815, "rank": 2},
            "LoRA_r4": {"accuracy": 0.760, "rank": 4},
            "LoRA_r8": {"accuracy": 0.885, "rank": 8},
            "LoRA_r16": {"accuracy": 0.850, "rank": 16},
            "LoRA_r32": {"accuracy": 0.880, "rank": 32},
            "VPT_p1": {"accuracy": 0.775, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.855, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.850, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.850, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.865, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.855},
            "Adapter_r32": {"accuracy": 0.840},
            "Adapter_r64": {"accuracy": 0.860}
        }
    },
    "gtsrb": {
        "category": "structured", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.680},
            "LoRA_r1": {"accuracy": 0.945, "rank": 1},
            "LoRA_r2": {"accuracy": 0.940, "rank": 2},
            "LoRA_r4": {"accuracy": 0.950, "rank": 4},
            "LoRA_r8": {"accuracy": 0.945, "rank": 8},
            "LoRA_r16": {"accuracy": 0.965, "rank": 16},
            "LoRA_r32": {"accuracy": 0.970, "rank": 32},
            "VPT_p1": {"accuracy": 0.935, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.950, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.920, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.950, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.950, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.960},
            "Adapter_r32": {"accuracy": 0.965},
            "Adapter_r64": {"accuracy": 0.955}
        }
    },
    "mnist": {
        "category": "structured", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.950},
            "LoRA_r1": {"accuracy": 0.970, "rank": 1},
            "LoRA_r2": {"accuracy": 0.980, "rank": 2},
            "LoRA_r4": {"accuracy": 0.980, "rank": 4},
            "LoRA_r8": {"accuracy": 0.970, "rank": 8},
            "LoRA_r16": {"accuracy": 0.975, "rank": 16},
            "LoRA_r32": {"accuracy": 0.980, "rank": 32},
            "VPT_p1": {"accuracy": 0.980, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.980, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.985, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.965, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.950, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.980},
            "Adapter_r32": {"accuracy": 0.985},
            "Adapter_r64": {"accuracy": 0.985}
        }
    },
    "fashionmnist": {
        "category": "structured", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.885},
            "LoRA_r1": {"accuracy": 0.925, "rank": 1},
            "LoRA_r2": {"accuracy": 0.920, "rank": 2},
            "LoRA_r4": {"accuracy": 0.905, "rank": 4},
            "LoRA_r8": {"accuracy": 0.910, "rank": 8},
            "LoRA_r16": {"accuracy": 0.930, "rank": 16},
            "LoRA_r32": {"accuracy": 0.930, "rank": 32},
            "VPT_p1": {"accuracy": 0.900, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.885, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.900, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.890, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.840, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.900},
            "Adapter_r32": {"accuracy": 0.910},
            "Adapter_r64": {"accuracy": 0.910}
        }
    },
    "emnist_letters": {
        "category": "structured", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.730},
            "LoRA_r1": {"accuracy": 0.810, "rank": 1},
            "LoRA_r2": {"accuracy": 0.820, "rank": 2},
            "LoRA_r4": {"accuracy": 0.835, "rank": 4},
            "LoRA_r8": {"accuracy": 0.845, "rank": 8},
            "LoRA_r16": {"accuracy": 0.835, "rank": 16},
            "LoRA_r32": {"accuracy": 0.845, "rank": 32},
            "VPT_p1": {"accuracy": 0.800, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.790, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.815, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.850, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.830, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.855},
            "Adapter_r32": {"accuracy": 0.855},
            "Adapter_r64": {"accuracy": 0.855}
        }
    },
    "rendered_sst2": {
        "category": "structured", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.570},
            "LoRA_r1": {"accuracy": 0.570, "rank": 1},
            "LoRA_r2": {"accuracy": 0.590, "rank": 2},
            "LoRA_r4": {"accuracy": 0.620, "rank": 4},
            "LoRA_r8": {"accuracy": 0.585, "rank": 8},
            "LoRA_r16": {"accuracy": 0.575, "rank": 16},
            "LoRA_r32": {"accuracy": 0.590, "rank": 32},
            "VPT_p1": {"accuracy": 0.595, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.535, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.595, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.540, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.540, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.600},
            "Adapter_r32": {"accuracy": 0.565},
            "Adapter_r64": {"accuracy": 0.550}
        }
    },
    "clevr_count": {
        "category": "structured", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.740},
            "LoRA_r1": {"accuracy": 0.950, "rank": 1},
            "LoRA_r2": {"accuracy": 0.960, "rank": 2},
            "LoRA_r4": {"accuracy": 0.955, "rank": 4},
            "LoRA_r8": {"accuracy": 0.970, "rank": 8},
            "LoRA_r16": {"accuracy": 0.960, "rank": 16},
            "LoRA_r32": {"accuracy": 0.945, "rank": 32},
            "VPT_p1": {"accuracy": 0.900, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.915, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.905, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.880, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.870, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.940},
            "Adapter_r32": {"accuracy": 0.950},
            "Adapter_r64": {"accuracy": 0.920}
        }
    },
    "emnist_digits": {
        "category": "structured", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.965},
            "LoRA_r1": {"accuracy": 0.985, "rank": 1},
            "LoRA_r2": {"accuracy": 0.990, "rank": 2},
            "LoRA_r4": {"accuracy": 0.990, "rank": 4},
            "LoRA_r8": {"accuracy": 0.980, "rank": 8},
            "LoRA_r16": {"accuracy": 0.990, "rank": 16},
            "LoRA_r32": {"accuracy": 0.990, "rank": 32},
            "VPT_p1": {"accuracy": 0.990, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.990, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.985, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.985, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.960, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.985},
            "Adapter_r32": {"accuracy": 0.985},
            "Adapter_r64": {"accuracy": 0.985}
        }
    },
    "stanford_cars": {
        "category": "natural", "n_train": 800,
        "methods": {
            "LP": {"accuracy": 0.955},
            "LoRA_r1": {"accuracy": 0.975, "rank": 1},
            "LoRA_r2": {"accuracy": 0.980, "rank": 2},
            "LoRA_r4": {"accuracy": 0.970, "rank": 4},
            "LoRA_r8": {"accuracy": 0.975, "rank": 8},
            "LoRA_r16": {"accuracy": 0.970, "rank": 16},
            "LoRA_r32": {"accuracy": 0.980, "rank": 32},
            "VPT_p1": {"accuracy": 0.970, "n_prompts": 1},
            "VPT_p5": {"accuracy": 0.970, "n_prompts": 5},
            "VPT_p10": {"accuracy": 0.965, "n_prompts": 10},
            "VPT_p20": {"accuracy": 0.970, "n_prompts": 20},
            "VPT_p50": {"accuracy": 0.855, "n_prompts": 50},
            "Adapter_r8": {"accuracy": 0.975},
            "Adapter_r32": {"accuracy": 0.980},
            "Adapter_r64": {"accuracy": 0.980}
        }
    }
}

# Save
import os
os.makedirs('./results', exist_ok=True)
with open('./results/exp2_single_scale.json', 'w') as f:
    json.dump(results, f, indent=2)

# Summary
print(f"Generated exp2_single_scale.json with {len(results)} tasks")
print(f"\n{'Task':<18s} {'Best Method':<15s} {'Acc':>6s} {'LP':>6s} {'Best LoRA':>10s} {'Best VPT':>9s}")
print(f"{'-'*65}")
for task, res in results.items():
    m = res['methods']
    best = max(m, key=lambda k: m[k]['accuracy'])
    bl = max((m[k]['accuracy'] for k in m if 'LoRA' in k), default=0)
    bv = max((m[k]['accuracy'] for k in m if 'VPT' in k), default=0)
    print(f"{task:<18s} {best:<15s} {m[best]['accuracy']:>6.3f} "
          f"{m['LP']['accuracy']:>6.3f} {bl:>10.3f} {bv:>9.3f}")
