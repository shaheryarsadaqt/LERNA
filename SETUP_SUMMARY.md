# Phase 1.2/1.3 Environment Setup Summary

**Server:** nanhangproj@219.223.251.100 (dgx01)  
**Date:** 2026-07-19  
**Status:** READY

---

## Environment: `lerna`

**Location:** `/new_raid/nanhangproj/conda_envs/lerna`

### System Specs
- **Python:** 3.10.12
- **OS:** Linux (Ubuntu-based)
- **GPUs:** 8x Tesla V100-SXM2-32GB (all healthy)
- **CUDA:** 11.8 (available)
- **Driver:** 525.105.17

### Installed Packages

| Package | Version | Required |
|---|---|---|
| torch | 2.0.1+cu118 | ✓ |
| transformers | 4.48.3 | ✓ |
| datasets | 4.8.4 | ✓ |
| evaluate | 0.4.6 | ✓ |
| huggingface_hub | 0.36.2 | ✓ |
| safetensors | 0.8.0 | ✓ |
| scipy | 1.15.2 | ✓ |
| scikit-learn | 1.7.2 | ✓ |
| numpy | 1.26.4 | ✓ |
| pandas | 2.3.3 | ✓ |
| tqdm | 4.68.3 | ✓ |
| accelerate | 1.14.0 | ✓ |
| deepspeed | 0.10.3 | ✓ |
| wandb | 0.28.0 | ✓ |
| matplotlib | 3.9.1 | ✓ |
| seaborn | 0.13.2 | ✓ |

### GPU Health

All 8 GPUs are healthy and accessible:
```
0, 00000000:06:00.0, Tesla V100-SXM2-32GB, 40.60 W, 37°C
1, 00000000:07:00.0, Tesla V100-SXM2-32GB, 41.08 W, 38°C
2, 00000000:0A:00.0, Tesla V100-SXM2-32GB, 41.08 W, 39°C
3, 00000000:0B:00.0, Tesla V100-SXM2-32GB, 42.08 W, 37°C
4, 00000000:85:00.0, Tesla V100-SXM2-32GB, 41.59 W, 38°C
5, 00000000:86:00.0, Tesla V100-SXM2-32GB, 40.12 W, 38°C
6, 00000000:89:00.0, Tesla V100-SXM2-32GB, 40.63 W, 39°C
7, 00000000:8A:00.0, Tesla V100-SXM2-32GB, 41.57 W, 36°C
```

### Verified Functionality

- ✓ All core LERNA modules import successfully
- ✓ CUDA enumeration: 8 GPUs detected
- ✓ Tensor operations on GPU: verified
- ✓ Package versions match DGX `torch_cuda` requirements

### Disk Space

| Mount | Total | Used | Available |
|---|---|---|---|
| `/dev/sda2` (home) | 439G | 405G | 11G (98%) |
| `/dev/sdc` (`/new_raid`) | 15T | 14T | 145G (99%) |

**Note:** Full 800-run Phase 1.2 matrix requires 150+ GB. Current `/dev/sdc` has 145G free.

### Usage

Activate the environment:
```bash
conda activate lerna
```

Run experiments:
```bash
cd /home/nanhangproj/LERNA
export HF_HOME="$HOME/ettin_hf_bundle"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0
python3 scripts/run_phase1_2_ettin.py --mode run ...
```

### Next Steps

1. Build or transfer `ettin_hf_bundle` to this server
2. Generate official manifest
3. Run smoke tests
4. Begin Phase 1.2 experiments (requires 150+ GB free space)
