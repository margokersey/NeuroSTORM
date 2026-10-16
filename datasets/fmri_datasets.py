import os
import pickle
import torch
from torch.utils.data import Dataset, IterableDataset
import torch.nn.functional as F

import numpy as np
import random
import math
import pandas as pd
from collections import OrderedDict


def pad_to_96(y):
    background_value = y.flatten()[0]
    y = y.permute(0,4,1,2,3)
    pad_1 = 96 - y.shape[-1]
    pad_2 = 96 - y.shape[-2]
    pad_3 = 96 - y.shape[-3]
    y = torch.nn.functional.pad(y, (math.ceil(pad_1/2), math.floor(pad_1/2), math.ceil(pad_2/2), math.floor(pad_2/2), math.ceil(pad_3/2), math.floor(pad_3/2)), value=background_value)[:,:,:,:,:]
    y = y.permute(0,2,3,4,1)

    return y

import torch
import torch.nn.functional as F

def resize_volume(y, target_size):
    """
    y: 5D tensor [B, H, W, D, T]
    target_size: 4d tuple/list (H', W', D', T)
    """
    current_size = y.shape

    if len(current_size) != 5:
        raise ValueError("Input y must be a 5-dimensional tensor.")
    if len(target_size) != 4:
        raise ValueError("Target size must be a tuple or list of length 4.")

    if current_size[-1] != target_size[-1]:
        raise ValueError(f"y's last dimension {current_size[-1]} and target_size's last dimension {target_size[-1]} must match.")
        
    if current_size[1:4] != tuple(target_size[:3]):
        resized_y = torch.empty(
            (current_size[0],) + tuple(target_size),
            dtype=y.dtype, device=y.device
        )

        for i in range(current_size[0]):
            for t in range(current_size[-1]):
                data_tensor = y[i, :, :, :, t]

                original_dtype = data_tensor.dtype
                resized_tensor = F.interpolate(data_tensor.float().unsqueeze(0).unsqueeze(0), size=tuple(target_size[:3]), mode='trilinear', align_corners=False).squeeze(0).squeeze(0).to(original_dtype)
                resized_y[i, :, :, :, t] = resized_tensor
        return resized_y
    else:
        return y


class BaseDataset(Dataset):
    # Upper bound on per-worker int8 blobs held in RAM at once.
    # Each blob is ~hundreds of MB; keep this small so 16 workers don't
    # blow up host memory (cache_max * num_workers * blob_size).
    _blob_cache_max = 8

    def __init__(self, **kwargs):
        super().__init__()
        self.register_args(**kwargs)
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(round(self.stride_between_seq * self.sample_duration), 1)
        self._data_blobs = OrderedDict()
        self._format_cache = {}
        self._num_frames_cache = {}
        self.data = self._set_data(self.root, self.subject_dict)

    def __getstate__(self):
        """Drop mmap'd blobs before pickling (e.g. for DataLoader workers)."""
        state = self.__dict__.copy()
        state['_data_blobs'] = OrderedDict()
        return state

    def register_args(self,**kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.kwargs = kwargs

    def _detect_format(self, subject_path):
        """Detect storage layout for a subject directory.

        Returns:
            'blob'   : new int8 single-file format (data.pt with mmap)
            'frames' : legacy per-frame float16 .pt files (frame_*.pt)
        """
        fmt = self._format_cache.get(subject_path)
        if fmt is None:
            if os.path.isfile(os.path.join(subject_path, 'data.pt')):
                fmt = 'blob'
            else:
                fmt = 'frames'
            self._format_cache[subject_path] = fmt
        return fmt

    @staticmethod
    def _load_blob_file(blob_path, mmap):
        """Load a trusted preprocessed data.pt blob across PyTorch versions."""
        try:
            return torch.load(blob_path, mmap=mmap, weights_only=True)
        except pickle.UnpicklingError:
            # Older preprocessing wrote numpy scalar metadata that PyTorch 2.6's
            # weights_only loader rejects. These blobs are generated locally.
            return torch.load(blob_path, mmap=mmap, weights_only=False)

    def _get_blob(self, subject_path):
        """Lazily load a subject's data.pt; LRU-evict older entries.

        mmap=True is the default and cheap, but some sshfs/NFS backends deny
        PROT_READ mappings (torch.load surfaces this as `Operation not permitted`).
        On that error we fall back to a plain read for that subject. Set
        NEUROSTORM_BLOB_MMAP=0 to skip the mmap attempt entirely.
        """
        blob = self._data_blobs.get(subject_path)
        if blob is not None:
            self._data_blobs.move_to_end(subject_path)
            return blob
        blob_path = os.path.join(subject_path, 'data.pt')
        prefer_mmap = os.environ.get("NEUROSTORM_BLOB_MMAP", "1") != "0"
        if prefer_mmap:
            try:
                blob = self._load_blob_file(blob_path, mmap=True)
            except RuntimeError as e:
                if "Operation not permitted" not in str(e):
                    raise
                blob = self._load_blob_file(blob_path, mmap=False)
        else:
            blob = self._load_blob_file(blob_path, mmap=False)
        self._data_blobs[subject_path] = blob
        while len(self._data_blobs) > self._blob_cache_max:
            self._data_blobs.popitem(last=False)
        return blob

    _frame_counts_json_cache = {}

    @classmethod
    def _load_frame_counts_json(cls, dataset_root):
        if dataset_root not in cls._frame_counts_json_cache:
            json_path = os.path.join(dataset_root, 'frame_counts.json')
            if os.path.isfile(json_path):
                import json
                with open(json_path) as f:
                    cls._frame_counts_json_cache[dataset_root] = json.load(f)
            else:
                cls._frame_counts_json_cache[dataset_root] = None
        return cls._frame_counts_json_cache[dataset_root]

    def _count_frames(self, subject_path):
        cached = self._num_frames_cache.get(subject_path)
        if cached is not None:
            return cached
        dataset_root = os.path.dirname(os.path.dirname(subject_path))
        fc_json = self._load_frame_counts_json(dataset_root)
        if fc_json is not None:
            subj_name = os.path.basename(subject_path)
            n = fc_json.get(subj_name, 0)
            self._num_frames_cache[subject_path] = n
            return n
        if self._detect_format(subject_path) == 'blob':
            peek_path = os.path.join(subject_path, 'data.pt')
            try:
                peek = self._load_blob_file(peek_path, mmap=True)
            except RuntimeError as e:
                if "Operation not permitted" not in str(e):
                    self._num_frames_cache[subject_path] = 0
                    return 0
                peek = self._load_blob_file(peek_path, mmap=False)
            n = int(peek['num_frames'])
            del peek
        else:
            n = len([f for f in os.listdir(subject_path)
                     if f.startswith('frame_') and f.endswith('.pt')])
        self._num_frames_cache[subject_path] = n
        return n

    def _load_clip(self, subject_path, indices):
        """Load the requested time indices and return float32 [1, H, W, D, T']."""
        if self._detect_format(subject_path) == 'blob':
            blob = self._get_blob(subject_path)
            frames = blob['frames']  # int8, [T, H, W, D]
            scale = float(blob['scale'])
            if isinstance(indices, range):
                clip = frames[indices.start:indices.stop:indices.step]
            else:
                clip = frames[torch.as_tensor(list(indices), dtype=torch.long)]
            clip = clip.to(torch.float32).mul_(scale)
            # [T', H, W, D] -> [1, H, W, D, T']
            return clip.permute(1, 2, 3, 0).unsqueeze(0)

        # legacy per-frame format: frame_i.pt is float16 [H, W, D, 1]
        idx_list = list(indices)
        parts = []
        for i in idx_list:
            frame = torch.load(os.path.join(subject_path, f'frame_{i}.pt'),
                               weights_only=True)
            parts.append(frame.to(torch.float32))
        clip = torch.cat(parts, dim=3)        # [H, W, D, T']
        return clip.unsqueeze(0)              # [1, H, W, D, T']

    def _maybe_append_voxel_norm(self, clip, subject_path):
        if not self.with_voxel_norm:
            return clip
        vm = torch.load(os.path.join(subject_path, 'voxel_mean.pt'), weights_only=True)
        vs = torch.load(os.path.join(subject_path, 'voxel_std.pt'), weights_only=True)
        # voxel_mean/std saved as [H, W, D, 1]; broadcast batch dim and concat along T
        if vm.ndim == 4:
            vm = vm.unsqueeze(0)
            vs = vs.unsqueeze(0)
        return torch.cat([clip, vm.to(clip.dtype), vs.to(clip.dtype)], dim=4)

    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None):
        if self.contrastive or self.mae:
            num_frames = self._count_frames(subject_path)

            if self.shuffle_time_sequence:
                indices = random.sample(range(num_frames), sample_duration // self.stride_within_seq)
            else:
                indices = range(start_frame, start_frame + sample_duration, self.stride_within_seq)
            y = self._load_clip(subject_path, indices)
            y = self._maybe_append_voxel_norm(y, subject_path)

            if self.mae:
                random_y = torch.zeros(1)
            else:
                # contrastive: pick a non-overlapping second clip
                full_range = np.arange(0, num_frames - sample_duration + 1)
                exclude_range = np.arange(start_frame - sample_duration, start_frame + sample_duration)
                available_choices = np.setdiff1d(full_range, exclude_range)
                if available_choices.size == 0:
                    # subject too short to fit two non-overlapping clips — fall back to any valid start
                    available_choices = full_range
                random_start_frame = int(np.random.choice(available_choices, size=1, replace=False)[0])
                rand_indices = range(random_start_frame, random_start_frame + sample_duration, self.stride_within_seq)
                random_y = self._load_clip(subject_path, rand_indices)
                random_y = self._maybe_append_voxel_norm(random_y, subject_path)

            return (y, random_y)

        # supervised path
        if self.shuffle_time_sequence:
            indices = random.sample(range(num_frames), sample_duration // self.stride_within_seq)
        else:
            indices = range(start_frame, start_frame + sample_duration, self.stride_within_seq)
        y = self._load_clip(subject_path, indices)
        y = self._maybe_append_voxel_norm(y, subject_path)
        return y

    def __len__(self):
        return  len(self.data)

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, _stride, num_frames, target, sex = self.data[index]
        sequence_length = self.sample_duration

        if self.contrastive or self.mae:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)
            y = pad_to_96(y)
            y = resize_volume(y, self.img_size)

            if self.contrastive:
                rand_y = pad_to_96(rand_y)
                rand_y = resize_volume(rand_y, self.img_size)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }
        else:   
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)
            y = pad_to_96(y)
            y = resize_volume(y, self.img_size)

            return {
                "fmri_sequence": y,
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            }

    def _set_data(self, root, subject_dict):
        raise NotImplementedError("Required function")
 

class HCP1200(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        for i, subject in enumerate(subject_dict):
            sex,target = subject_dict[subject]
            subject_path = os.path.join(img_root, subject)
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        return data


class ABCD(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            # subject_name = subject[4:]
            
            subject_path = os.path.join(img_root, subject_name)

            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class Cobre(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, subject_name)
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class ADHD200(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, '{}'.format(subject_name))
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data
        

class UCLA(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, '{}'.format(subject_name))
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class HCPEP(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, '{}'.format(subject_name))
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class GOD(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, '{}'.format(subject_name))
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class UKB(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            # subject_name = subject[4:]
            
            subject_path = os.path.join(img_root, subject_name)

            num_frames = self._count_frames(subject_path)
            if num_frames < self.sample_duration:
                continue
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data
    

class HCPTASK(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, subject_name)

            num_frames = self._count_frames(subject_path)
            if num_frames < self.sample_duration:
                continue
            session_duration = num_frames - self.sample_duration + 1

            # we only use first n frames for task fMRI
            data_tuple = (i, subject_name, subject_path, 0, self.stride, num_frames, target, sex)
            data.append(data_tuple)

        if self.train:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class MOVIE(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_tsv(self, path):
        df = pd.read_csv(path, sep='\t')
        result = []
        for _, row in df.iterrows():
            if row['trial_type'] == 'High_cal_food':
                label = 0
            elif row['trial_type'] == 'Low_cal_food':
                label = 1
            elif row['trial_type'] == 'non-food':
                label = 2
            else:
                raise ValueError(f"unexpected trial_type: {row['trial_type']}")
            entry = {
                'start': int(row['onset'] / 0.8),
                'end': int((row['onset'] + row['duration']) / 0.8) + 10,
                'label': label
            }
            result.append(entry)

        result[-1]['end'] -= 10

        return result

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject in enumerate(subject_dict):
            sex, target = subject_dict[subject]
            subject_path = os.path.join(img_root, subject)
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        # label_root = os.path.join(root, 'metadata')

        # for i, subject_name in enumerate(subject_dict):
        #     sex, target = subject_dict[subject_name]
        #     subject_path = os.path.join(img_root, subject_name)
        #     label_path = os.path.join(label_root, '{}-food_events.tsv'.format(subject_name[:-5]))
        #     label = self.process_tsv(label_path)

        #     num_frames = len(os.listdir(subject_path))
        #     if num_frames < self.stride:
        #         import ipdb; ipdb.set_trace()
        #     session_duration = num_frames - self.sample_duration + 1

        #     for j in range(len(label)):
        #         for start_frame in range(label[j]['start'], label[j]['end'], self.stride):
        #             if start_frame + self.stride >= num_frames:
        #                 continue

        #             data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, label[j]['label'], sex)
        #             data.append(data_tuple)
            
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


def assign_blocks_to_parts(blocks, fracs, seed=0):
    """Deterministic, class-stratified train/val/test assignment over a
    list of (start, end, label) blocks -- used by MichiganSampleSplit to
    split an individual subject's OWN blocks (for MICHIGAN_WITHIN /
    MICHIGAN_POOLED) instead of only whole-subject membership.

    Stratifying by label keeps every part class-balanced even with as few
    as 10 instances per class (Michigan's task6 has exactly 10 tennis /
    10 navigation / 10 squeeze blocks per subject). The assignment only
    depends on `blocks`, `fracs`, `seed` -- NOT on which subject is
    asking -- so e.g. "block #14 (a navigation instance)" is the same
    held-out test instance for every subject. That keeps every subject's
    train/val/test class balance identical and the whole pooled dataset
    directly comparable subject to subject.

    Returns `part_of_block`, a list parallel to `blocks`, with values in
    {"train", "val", "test"}.
    """
    import random
    by_label = {}
    for i, (_, _, lab) in enumerate(blocks):
        by_label.setdefault(lab, []).append(i)

    part_of_block = [None] * len(blocks)
    rng = random.Random(seed)
    for lab, idxs in sorted(by_label.items()):
        idxs = idxs[:]
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = min(int(round(n * fracs["train"])), n)
        n_val = min(int(round(n * fracs["val"])), n - n_train)
        train_idxs = idxs[:n_train]
        val_idxs = idxs[n_train:n_train + n_val]
        test_idxs = idxs[n_train + n_val:]
        for i in train_idxs:
            part_of_block[i] = "train"
        for i in val_idxs:
            part_of_block[i] = "val"
        for i in test_idxs:
            part_of_block[i] = "test"
    return part_of_block


class Michigan(BaseDataset):
    """task-imagery_run-1 block design: tennis / navigation / squeeze.

    Each 3-way label is assigned to a clip spanning the task block PLUS the
    rest period immediately following it (participants keep imagining the
    task for several seconds after being told to rest, so that trailing
    rest carries task-related signal). squeeze_imagery + squeeze_action are
    merged into a single 'squeeze' condition since they are one continuous
    behavioral block. Verified against timing_onehot_task1.tsv: every
    (task + following rest) block is 37-38 TR (29.6-30.4 s at TR=0.8s), so a
    fixed 32-frame window taken from the start of each (trimmed) block fits
    every block with margin to spare.
    """

    LABEL_MAP = {'tennis': 0, 'navigation': 1, 'squeeze': 2}
    TRIM_START = 2  # drop ~1.6s at block onset (still reflects the previous condition)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _blocks_from_tsv(self, tsv_path, trim_start=None):
        """Return [(start_frame, end_frame, task_name), ...] in TR units,
        where each block = one task run merged with its immediately
        following rest run (if any), trimmed by `trim_start` at the front.

        trim_start defaults to None so subclasses that override TRIM_START
        (e.g. MichiganFullBlock, which sets TRIM_START = 0) actually take
        effect -- a default bound at class-definition time would silently
        ignore any subclass override.
        """
        if trim_start is None:
            trim_start = self.TRIM_START
        df = pd.read_csv(tsv_path, sep='\t')
        df['squeeze'] = df['squeeze_imagery'] | df['squeeze_action']
        cols = ['rest', 'squeeze', 'navigation', 'tennis']
        labels = df[cols].idxmax(axis=1).tolist()

        # run-length encode the full label sequence (rest included)
        runs = []
        prev, start = None, 0
        for i, lab in enumerate(labels + [None]):
            if lab != prev:
                if prev is not None:
                    runs.append((start, i, prev))
                start, prev = i, lab

        blocks = []
        n = len(runs)
        for idx, (s, e, lab) in enumerate(runs):
            if lab == 'rest':
                continue
            end = e
            if idx + 1 < n and runs[idx + 1][2] == 'rest':
                end = runs[idx + 1][1]  # absorb the following rest run
            blocks.append((s + trim_start, end, lab))
        return blocks

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        tsv_path = os.path.join(root, 'metadata', 'timing_onehot_task1.tsv')
        blocks = self._blocks_from_tsv(tsv_path)
        # fmri_reid: classify WHICH SUBJECT a window came from instead of
        # which task block it was -- the label is a GLOBAL contiguous
        # subject id assigned once in data_module.py's MICHIGAN branch
        # (subject_dict[subject][1]), BEFORE this dict gets split into
        # train/val/test, so the same id always means the same subject
        # across all three parts. Every other task_name keeps the original
        # task-block label from LABEL_MAP, unchanged.
        is_reid = getattr(self, 'task_name', None) == 'fmri_reid'

        for i, subject in enumerate(subject_dict):
            subject_path = os.path.join(img_root, subject)
            num_frames = self._count_frames(subject_path)
            for (start, end, task_name) in blocks:
                if start + self.sample_duration > end or start + self.sample_duration > num_frames:
                    continue
                label = subject_dict[subject][1] if is_reid else self.LABEL_MAP[task_name]
                data_tuple = (i, subject, subject_path, start, self.stride, num_frames, label, 0)
                data.append(data_tuple)

        if self.train:
            self.target_values = np.array([t[6] for t in data]).reshape(-1, 1)
        return data


class MichiganFullBlock(Michigan):
    """Same task-imagery_run-1 block parsing as Michigan, but with no
    onset trim -- TRIM_START = 0 means _blocks_from_tsv returns each
    block's FULL (task + trailing rest) span starting right at the
    TSV-labeled onset, instead of the 2-TR-trimmed clips Michigan uses.

    This does NOT require regenerating data.pt: the underlying recordings
    already contain every frame; only the block boundaries used at load
    time change. _set_data (inherited unchanged) still requires
    `start + self.sample_duration <= end`, so --sequence_length_override
    (and the matching img_size_override) must be set to a value that fits
    inside every block's full span at trim_start=0. Use
    window_sweep_linear_probe.py's max_window_for_trim, run with
    --trims 0, to read off that value for this dataset.
    """
    TRIM_START = 0


class MichiganSampleSplit(Michigan):
    """Same task-imagery_run-1 block design as Michigan (10 tennis / 10
    navigation / 10 squeeze blocks per subject, from
    timing_onehot_task1.tsv), but splits an individual subject's OWN
    blocks across train/val/test instead of assigning a whole subject to
    exactly one of the three sets. This is what makes both of the
    following possible without touching any data:

      - MICHIGAN_WITHIN:  one model per subject, trained and tested only
        on that subject's own blocks (pure within-subject decoding).
      - MICHIGAN_POOLED:  one model trained on ALL subjects pooled
        together, where each subject contributes some blocks to train
        and other blocks to val/test (the "mixed" condition -- subjects
        can appear in both train and test, unlike Michigan's
        between-subject 5-fold CV).

    subject_dict keys must be virtual "<real_subject>::<part>" strings
    (part in {train, val, test}), as produced by data_module.py's
    MICHIGAN_WITHIN / MICHIGAN_POOLED subject-dict branch -- this is the
    only way to get the SAME subject to land in more than one of
    train_dict/val_dict/test_dict, since _build_single_dataset partitions
    subject_dict purely by key membership in the split file's
    train/val/test subject lists.

    _set_data resolves each virtual key back to its real subject folder
    for file I/O, computes the same 30-block design Michigan does, and
    keeps only the blocks assign_blocks_to_parts assigned to that key's
    `part`. See BLOCK_SPLIT_FRACS / BLOCK_SPLIT_SEED to change the
    per-subject train/val/test ratio -- the same assignment is used for
    every subject (see assign_blocks_to_parts's docstring), so every part
    stays class-balanced and directly comparable subject to subject.

    Default 0.7/0.1/0.2 split -> per subject: 21 train (7/class),
    3 val (1/class), 6 test (2/class) -- test is kept at 2/class rather
    than 1/class since a single-subject "fold" only has 30 blocks total
    to begin with and a 1-sample-per-class test set is too noisy to read
    (only 4 possible accuracy values). Still very little data for a
    ~5M-parameter full fine-tune per subject (MICHIGAN_WITHIN) -- treat
    any single subject's number as noisy and look at the distribution
    across all 26 subjects (e.g. via compare_conditions.py, treating each
    subject as a "fold") rather than trusting one subject's result alone.
    """
    BLOCK_SPLIT_FRACS = {"train": 0.7, "val": 0.1, "test": 0.2}
    BLOCK_SPLIT_SEED = 0
    TRIM_START = 0

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        tsv_path = os.path.join(root, 'metadata', 'timing_onehot_task1.tsv')
        all_blocks = self._blocks_from_tsv(tsv_path)
        part_of_block = assign_blocks_to_parts(
            all_blocks, self.BLOCK_SPLIT_FRACS, seed=self.BLOCK_SPLIT_SEED)

        for i, key in enumerate(subject_dict):
            if '::' not in key:
                raise ValueError(
                    "MichiganSampleSplit expects '<subject>::<part>' subject_dict "
                    "keys (see data_module.py's MICHIGAN_WITHIN/MICHIGAN_POOLED "
                    "branch), got %r" % (key,))
            real_subject, part = key.rsplit('::', 1)
            subject_path = os.path.join(img_root, real_subject)
            num_frames = self._count_frames(subject_path)
            for bi, (start, end, task_name) in enumerate(all_blocks):
                if part_of_block[bi] != part:
                    continue
                if start + self.sample_duration > end or start + self.sample_duration > num_frames:
                    continue
                label = self.LABEL_MAP[task_name]
                data_tuple = (i, key, subject_path, start, self.stride, num_frames, label, 0)
                data.append(data_tuple)

        if self.train:
            self.target_values = np.array([t[6] for t in data]).reshape(-1, 1)
        return data


class MichiganGender(BaseDataset):
    """Subject-level binary sex classification on the same task-imagery_run-1
    recordings as Michigan/MichiganFullBlock, using participants.tsv
    (columns: participant_id, age, sex, lor_esc, original_id) instead of
    block-level timing_onehot_task1.tsv labels.

    Slides a fixed sample_duration window across each subject's whole scan
    (same windowing convention as HCP1200/ABCD's subject-level phenotype
    prediction), assigning every window that subject's sex label. No TSV
    block parsing -- every frame of the scan is equally valid for this task.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject in enumerate(subject_dict):
            sex, target = subject_dict[subject]
            subject_path = os.path.join(img_root, subject)
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        if self.train:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        return data


class TransDiag(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject in enumerate(subject_dict):
            sex, target = subject_dict[subject]
            subject_path = os.path.join(img_root, subject)
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        if self.train:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class ABIDE(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, subject_name)
            num_frames = self._count_frames(subject_path)
            if num_frames < self.sample_duration:
                continue
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        if self.train:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data
