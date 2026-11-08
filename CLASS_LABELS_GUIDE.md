# Class Labels Guide for NeuroSTORM Fine-Tuning (Michigan Dataset)

## TL;DR: The 3-Way Task Labels

The fine-tuning task (`task6` / `imagery_task_classification`) is a **3-way classification** task:

| Class ID | Label | Brain Activity |
|----------|-------|----------------|
| 0 | **tennis** | Motor imagery: imagining playing tennis |
| 1 | **navigation** | Spatial navigation imagery |
| 2 | **squeeze** | Motor imagery: imagining squeezing with hands |

## Where Do the Labels Come From?

### 1. **Source: Timing TSV File**
```
Location: $DATA_ROOT/FIXED_xcp_nosmooth_task-imagery_run-1/metadata/timing_onehot_task1.tsv
Current:  /nfs/turbo/umms-huanglab/mkersey/Michigan_Dataset_1_for_NeuroSTORM/FIXED_xcp_nosmooth_task-imagery_run-1/metadata/timing_onehot_task1.tsv
```

**Format** (one-hot encoded per TR = 0.8s):
```
rest    squeeze_imagery    squeeze_action    navigation    tennis
1       0                  0                 0             0         ← rest period
1       0                  0                 0             0
...
0       1                  0                 0             0         ← squeeze_imagery
0       1                  0                 0             0
...
0       0                  1                 0             0         ← squeeze_action
0       0                  1                 0             0
...
0       0                  0                 1             0         ← navigation
0       0                  0                 1             0
...
0       0                  0                 0             1         ← tennis
0       0                  0                 0             1
...
```

**File contents:**
- **Rows**: One per TR (~313 rows for a ~4 minute fMRI run at TR=0.8s)
- **Columns**: 5 binary indicators (rest, squeeze_imagery, squeeze_action, navigation, tennis)
- **Exactly one column is 1 per row** (mutually exclusive)
- **Each class lasts multiple TRs** (motor imagery blocks: ~25-30s = 31-38 TRs each)

### 2. **Label Processing: In Code**

#### File: [datasets/fmri_datasets.py](/home/mkersey/code/NeuroSTORM/datasets/fmri_datasets.py)

**Michigan dataset class:**
```python
class Michigan(BaseDataset):
    """task-imagery_run-1 block design: tennis / navigation / squeeze.
    
    Each 3-way label is assigned to a clip spanning the task block PLUS the
    rest period immediately following it (participants keep imagining the
    task for several seconds after being told to rest, so that trailing
    rest carries task-related signal). squeeze_imagery + squeeze_action are
    merged into a single 'squeeze' condition since they are one continuous
    behavioral block.
    """
    
    LABEL_MAP = {'tennis': 0, 'navigation': 1, 'squeeze': 2}
    TRIM_START = 2  # drop ~1.6s at block onset
```

**Label mapping:**
```python
# From timing_onehot_task1.tsv labels → integer class IDs
LABEL_MAP = {
    'tennis': 0,
    'navigation': 1, 
    'squeeze': 2
}
```

**Block extraction process:**
```python
def _blocks_from_tsv(self, tsv_path, trim_start=None):
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Merge squeeze_imagery + squeeze_action into one 'squeeze' class
    df['squeeze'] = df['squeeze_imagery'] | df['squeeze_action']
    
    # Get dominant class per row (which one is 1)
    cols = ['rest', 'squeeze', 'navigation', 'tennis']
    labels = df[cols].idxmax(axis=1).tolist()
    
    # Run-length encode: find contiguous blocks of same label
    # Each block = (start_frame, end_frame, task_name)
    # Then extract "task + following rest" spans
    # (because subjects keep imagining during rest)
    
    blocks = _blocks_from_runs(labels, trim_start=2)
    # Returns: [(start_TR, end_TR, 'tennis'), (start_TR, end_TR, 'navigation'), ...]
    
    return blocks
```

**Label assignment during training:**
```python
def _set_data(self, root, subject_dict):
    # For each subject
    for subject in subjects:
        blocks = self._blocks_from_tsv(tsv_path)  # Load timing
        
        # For each task block
        for start, end, task_name in blocks:
            if task_name != 'rest':  # Skip pure rest blocks
                # Map task name to class ID
                label = self.LABEL_MAP[task_name]  # 0, 1, or 2
                
                # Extract sliding windows from [start, end)
                # Each window is 32 TRs (fits in every block)
                for window_start in range(start, end - 32):
                    data_tuple = (
                        sample_id, 
                        subject, 
                        subject_path,
                        window_start,           # TR start
                        32,                     # sequence_length (32 frames)
                        label,                  # ← CLASS ID (0, 1, or 2)
                        0
                    )
                    data.append(data_tuple)
    
    return data
```

## How Labels Flow to the Model

### Training Pipeline:

```
timing_onehot_task1.tsv (one-hot per TR)
    ↓
_blocks_from_tsv()  [extract task blocks]
    ↓
Block boundaries + task_name ('tennis', 'navigation', 'squeeze')
    ↓
LABEL_MAP[task_name]  [convert to integer: 0, 1, 2]
    ↓
Create sliding windows with labels
    ↓
DataLoader yields (img_tensor, label) pairs
    ↓
Model forward pass + loss calculation
    ↓
Lightning training loop
```

### Data Loading During Fine-tuning:

1. **Config file** (scripts/configs/datasets/michigan_xcpd_nosmooth_fullblock.yaml):
   ```yaml
   tasks:
     task6:
       downstream_task_id: 6
       task_name: imagery_task_classification
       downstream_task_type: classification
       num_classes: 3
   ```

2. **Fine-tuning script** runs:
   ```bash
   bash scripts/run_experiment.sh \
       --dataset michigan_xcpd_nosmooth_fullblock \
       --task task6 \
       ...
   ```

3. **Data module** loads splits (nas/splits/MICHIGAN/split_fixed_101.txt, etc.):
   - Selects which subjects are train/val/test for this fold
   - For each subject, loads blocks from timing_onehot_task1.tsv
   - Creates windows with labels

4. **Model** receives:
   - **Input**: fMRI volume (96×96×96×32 after preprocessing)
   - **Target**: integer label (0, 1, or 2)

## Complete Example: One Training Sample

### Scenario
- Subject: `sub-200`
- Timing file row 50-85: tennis class (1 in tennis column, 0 elsewhere)
- After TRIM_START=2, block starts at TR 52
- Extract window at TR 52-84 (32 TRs = 25.6 seconds)

### Processing Steps

```
1. Load timing_onehot_task1.tsv
   Row 50-85: tennis = 1, others = 0
   
2. Run-length encode:
   Detect contiguous tennis block: TR 50-85
   
3. Check for following rest:
   TR 85-92: rest = 1 (yes, rest follows)
   So block span = TR 50-92 (task + following rest)
   
4. Apply TRIM_START=2:
   Trimmed block = TR 52-92 (drop ~1.6s at onset)
   
5. Extract window at TR 52:
   Window = [52, 84) (32 frames)
   
6. Assign label:
   task_name = 'tennis'
   label = LABEL_MAP['tennis'] = 0
   
7. Create sample:
   (img_tensor [96,96,96,32], label=0)
   
8. Model training:
   forward(img_tensor) → logits [3]  (3 classes)
   loss = CrossEntropyLoss(logits, target=0)
   backward()
```

## Split Files: How Train/Val/Test Subjects Are Assigned

**File:** `nas/splits/MICHIGAN/split_fixed_101.txt` (for LOOCV fold 101)

**Format** (example):
```
train_subjects
sub-200
sub-201
sub-202
sub-203
sub-204
sub-205
sub-206
sub-207
sub-208
sub-209
sub-210
sub-211
sub-212
sub-213
sub-214
sub-215
sub-216
sub-217
sub-218
sub-219
sub-220
sub-221
sub-222
sub-223
val_subjects
sub-224
test_subjects
sub-225
```

**Meaning (LOOCV fold 101):**
- **train**: 24 subjects (sub-200 to sub-223)
- **val**: 1 subject (sub-224)
- **test (held-out)**: 1 subject (sub-225)

Each subject's timing comes from the SAME `timing_onehot_task1.tsv`, but train/val/test set is determined by which subjects are in which partition.

## Key Points for Understanding Fine-Tuning

### 1. **Labels are fixed per block**
- No per-sample variation within a task block
- All 32-frame windows from the tennis block (TR 50-92) have label=0
- All windows from navigation block have label=1, squeeze has label=2

### 2. **Balanced across subjects**
- Each subject has ~10 blocks of each class (30 total blocks)
- Training samples are stratified by class during split creation
- This ensures balanced representation in train/val/test

### 3. **No label leakage between train/val/test**
- Split is at **subject level**, not sample level
- If subject-X is in training, ALL their blocks are in training
- No within-subject block contamination across partitions

### 4. **LOOCV uses the same labels**
- All 26 LOOCV folds use the SAME `timing_onehot_task1.tsv`
- Only the subject partition differs (split_fixed_101.txt, ..., split_fixed_126.txt)
- Each fold: 22 train subjects, 3 val subjects, 1 test subject

## Verification: How to Check Labels in Your Experiment

### 1. **Check label distribution during training**
```python
# In your training loop or validation callback
from datasets.fmri_datasets import Michigan

michigan = Michigan(...)
target_values = michigan.target_values  # Shape: [N_samples, 1]

import numpy as np
unique, counts = np.unique(target_values, return_counts=True)
for label_id, count in zip(unique, counts):
    label_name = {0: 'tennis', 1: 'navigation', 2: 'squeeze'}[label_id]
    print(f"{label_name}: {count} samples")
```

### 2. **Inspect raw timing file**
```bash
# Count blocks per class
grep -E "^0.*1.*0.*0$" /nfs/turbo/.../timing_onehot_task1.tsv | wc -l  # tennis
grep -E "^0.*0.*1.*0$" /nfs/turbo/.../timing_onehot_task1.tsv | wc -l  # squeeze_imagery
grep -E "^0.*0.*0.*1$" /nfs/turbo/.../timing_onehot_task1.tsv | wc -l  # navigation
```

### 3. **Print sample from data loader**
```python
from data_module import MichiganDataModule
from torch.utils.data import DataLoader

dm = MichiganDataModule(...)
train_loader = DataLoader(dm.train_dataset, batch_size=1, shuffle=False)

for imgs, labels in train_loader:
    print(f"Image shape: {imgs.shape}, Label: {labels.item()}")
    label_names = {0: 'tennis', 1: 'navigation', 2: 'squeeze'}
    print(f"Class: {label_names[labels.item()]}")
    break
```

## Related Files Reference

| File | Purpose | Content |
|------|---------|---------|
| [datasets/fmri_datasets.py](/home/mkersey/code/NeuroSTORM/datasets/fmri_datasets.py) | Label loading, block extraction | Michigan class, LABEL_MAP, _blocks_from_tsv() |
| [scripts/configs/datasets/michigan_xcpd_nosmooth_fullblock.yaml](/home/mkersey/code/NeuroSTORM/scripts/configs/datasets/michigan_xcpd_nosmooth_fullblock.yaml) | Task/dataset config | num_classes: 3, task_name: imagery_task_classification |
| `/nfs/turbo/.../timing_onehot_task1.tsv` | Ground truth labels | One-hot per TR (rest, squeeze_imagery, squeeze_action, navigation, tennis) |
| `nas/splits/MICHIGAN/split_fixed_*.txt` | Train/val/test subject partition | One subject ID per line, grouped by partition |

## Summary

- **3 classes**: tennis (0), navigation (1), squeeze (2)
- **Source**: timing_onehot_task1.tsv (one-hot encoded per TR)
- **Processing**: Run-length encode → extract task blocks + following rest → trim 2 TRs → extract 32-frame windows
- **Assignment**: Each window gets the task name → converted to integer via LABEL_MAP
- **Splits**: Subject-level partitioning (train/val/test subjects from split files)
- **Fine-tuning**: Model sees 32-frame fMRI clips + class ID (0/1/2)
