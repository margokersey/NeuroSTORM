"""
Patch datasets/fmri_datasets.py to add a MichiganFullBlock dataset variant.

What this does (and does NOT do):
  - Does NOT touch, regenerate, or copy any data.pt files. The existing
    processed_output_sub_voxel_norm files already contain full, uncropped
    recordings; only the load-time block-boundary logic changes.
  - Fixes a latent bug: Michigan._blocks_from_tsv had `trim_start=TRIM_START`
    as its default, which binds the literal value 2 at class-definition
    time. A subclass overriding TRIM_START would silently NOT change this
    default. Fixed to `trim_start=None` with a runtime fallback to
    `self.TRIM_START`, so subclass overrides actually take effect.
  - Adds MichiganFullBlock(Michigan) with TRIM_START = 0, so each block's
    FULL (task + trailing rest) span is used, starting right at the TSV
    onset, instead of Michigan's 2-TR-trimmed clips.

Run from the repo root:
    python patch_michigan_full_block.py
"""
import io

TARGET = "datasets/fmri_datasets.py"

OLD_SIGNATURE = '''    def _blocks_from_tsv(self, tsv_path, trim_start=TRIM_START):
        """Return [(start_frame, end_frame, task_name), ...] in TR units,
        where each block = one task run merged with its immediately
        following rest run (if any), trimmed by `trim_start` at the front.
        """
        df = pd.read_csv(tsv_path, sep='\\t')'''

NEW_SIGNATURE = '''    def _blocks_from_tsv(self, tsv_path, trim_start=None):
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
        df = pd.read_csv(tsv_path, sep='\\t')'''

OLD_CLASS_BOUNDARY = '''        if self.train:
            self.target_values = np.array([t[6] for t in data]).reshape(-1, 1)
        return data


class TransDiag(BaseDataset):'''

NEW_CLASS_BOUNDARY = '''        if self.train:
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


class TransDiag(BaseDataset):'''


def main():
    with io.open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    assert src.count(OLD_SIGNATURE) == 1, (
        "Expected exactly one match for the _blocks_from_tsv signature block "
        "in %s -- file may already be patched or has changed unexpectedly." % TARGET
    )
    assert src.count(OLD_CLASS_BOUNDARY) == 1, (
        "Expected exactly one match for the Michigan/TransDiag class boundary "
        "in %s -- file may already be patched or has changed unexpectedly." % TARGET
    )

    src = src.replace(OLD_SIGNATURE, NEW_SIGNATURE)
    src = src.replace(OLD_CLASS_BOUNDARY, NEW_CLASS_BOUNDARY)

    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)

    print("Patched %s:" % TARGET)
    print("  - _blocks_from_tsv now resolves trim_start via self.TRIM_START at call time")
    print("  - added class MichiganFullBlock(Michigan) with TRIM_START = 0")


if __name__ == "__main__":
    main()
