# Class Labels Quick Reference

## The 3 Classes

| ID | Label | Task | Brain Region |
|---|---|---|---|
| **0** | **tennis** | Motor imagery: imagine playing tennis | Motor cortex (bilateral) |
| **1** | **navigation** | Spatial imagery: imagine walking through environment | Hippocampus, PPC |
| **2** | **squeeze** | Motor imagery: imagine squeezing hands | Motor cortex (bilateral) |

## Where the Labels Are

**File:** `$DATA_ROOT/FIXED_xcp_nosmooth_task-imagery_run-1/metadata/timing_onehot_task1.tsv`

**Current path:** `/nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1/metadata/timing_onehot_task1.tsv`

**Format:** One-hot encoded per TR (0.8 seconds)
```
rest    squeeze_imagery    squeeze_action    navigation    tennis
1       0                  0                 0             0
1       0                  0                 0             0
...
0       1                  0                 0             0     ← squeeze block
...
0       0                  0                 1             0     ← navigation block
...
0       0                  0                 0             1     ← tennis block
```

## How It Works

1. **Load** `timing_onehot_task1.tsv` → get one-hot per TR
2. **Extract** dominant class per row → `['rest', 'squeeze', 'navigation', 'tennis', ...]`
3. **Merge** `squeeze_imagery + squeeze_action` → single `'squeeze'` class
4. **Run-length encode** → find contiguous task blocks
5. **Extract** each task block + following rest period
6. **Trim** 2 TRs from start (~1.6 seconds, drop pre-task activity)
7. **Create windows** of 32 TRs each (25.6 seconds)
8. **Map** to integer: `'tennis'→0, 'navigation'→1, 'squeeze'→2`
9. **Train** model with (fMRI_volume, label) pairs

## During LOOCV Fine-Tuning

- **All 26 folds use the SAME `timing_onehot_task1.tsv`**
- Only the **subject partition** differs (split files)
- Each fold:
  - Train: 24 subjects (~240 blocks, ~80 per class)
  - Val: 1 subject (~30 blocks, ~10 per class)
  - Test (held-out): 1 subject (~30 blocks, ~10 per class)

## Code Location

- **Label definition:** [datasets/fmri_datasets.py](/home/mkersey/code/NeuroSTORM/datasets/fmri_datasets.py)
  ```python
  class Michigan(BaseDataset):
      LABEL_MAP = {'tennis': 0, 'navigation': 1, 'squeeze': 2}
  ```

- **Label extraction:** `Michigan._blocks_from_tsv()` method
  - Reads timing TSV
  - Parses task blocks
  - Applies trimming
  - Maps names to IDs

- **Config:** [scripts/configs/datasets/michigan_xcpd_nosmooth_fullblock.yaml](/home/mkersey/code/NeuroSTORM/scripts/configs/datasets/michigan_xcpd_nosmooth_fullblock.yaml)
  ```yaml
  tasks:
    task6:
      downstream_task_id: 6
      task_name: imagery_task_classification
      downstream_task_type: classification
      num_classes: 3
  ```

## Key Points

✓ **3-way classification** (not binary)  
✓ **Labels are block-level**, not frame-level (all windows from a block get same label)  
✓ **Balanced by class**: ~10 blocks per class per subject  
✓ **Balanced by subject**: all subjects included in train/val/test  
✓ **Subject-level split**: no within-subject contamination (if sub-X is train, ALL their blocks are train)  
✓ **Fixed across all folds**: timing_onehot_task1.tsv never changes, only subject partitions differ  

## Verify Labels in Your Experiment

```python
# Check label distribution
from datasets.fmri_datasets import Michigan
michigan = Michigan(...)
print(michigan.target_values)  # [N_samples, 1] array of 0s, 1s, 2s

# Count by class
import numpy as np
unique, counts = np.unique(michigan.target_values, return_counts=True)
for label_id, count in zip(unique, counts):
    names = {0: 'tennis', 1: 'navigation', 2: 'squeeze'}
    print(f"{names[label_id]}: {count} samples")
```

## See Also

- [CLASS_LABELS_GUIDE.md](/home/mkersey/code/NeuroSTORM/CLASS_LABELS_GUIDE.md) - Full detailed guide
- [datasets/fmri_datasets.py](/home/mkersey/code/NeuroSTORM/datasets/fmri_datasets.py) - Implementation
- `nas/splits/MICHIGAN/split_fixed_*.txt` - Subject partitions
