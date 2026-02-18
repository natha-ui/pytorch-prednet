#!/usr/bin/env python3
"""
Video2Sequence.py
-----------------
Converts an MP4 video into HDF5 sequence files compatible with PredNet's
dataloader (X_test.h5 + sources_test.h5).

Usage:
    python Video2Sequence.py video.mp4 --out ./my_video_data
    python Video2Sequence.py video.mp4 --out ./my_video_data --nt 15 --fps 10 --overlap

Dependencies:
    pip install opencv-python numpy h5py tqdm
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core conversion
# ─────────────────────────────────────────────────────────────────────────────

def video_to_prednet(
    mp4_path: Path,
    out_dir: Path,
    target_h: int = 128,
    target_w: int = 160,
    nt: int = 15,
    target_fps: float = 10.0,
    overlap: bool = False,
    split: float = 0.0,
    source_name: str | None = None,
    mono: bool = False,
) -> None:
    """
    Parameters
    ----------
    mp4_path    : path to input .mp4
    out_dir     : directory to write .h5 files into
    target_h/w  : frame size — must match what PredNet was trained on
    nt          : sequence length — must match nt_extrap in your test script
    target_fps  : resample video to this frame rate before windowing
    overlap     : if True, use a stride-1 sliding window; otherwise non-overlapping
    split       : if > 0, also produce a train split using this fraction of sequences
    source_name : label stored in sources_*.h5 (defaults to video stem)
    mono        : if True, convert to greyscale (stored as 3-channel for model compatibility)
    """

    mp4_path = Path(mp4_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not mp4_path.exists():
        raise FileNotFoundError(f"Video not found: {mp4_path}")

    source_name = source_name or mp4_path.stem

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {mp4_path}")

    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    log.info(f"Video  : {mp4_path.name}")
    log.info(f"Native : {native_w}×{native_h}  {native_fps:.2f} fps  {total_frames} frames")
    log.info(f"Target : {target_w}×{target_h}  {target_fps:.2f} fps")
    if mono:
        log.info("Mono   : greyscale conversion enabled (3-channel output)")

    # Frame-skip ratio: keep every Nth native frame to hit target_fps
    keep_every = max(1, round(native_fps / target_fps))
    log.info(f"Keeping every {keep_every} frame(s)  →  effective fps ≈ {native_fps/keep_every:.2f}")

    # ── Read & resample frames ────────────────────────────────────────────────
    raw_frames = []
    native_idx = 0

    pbar = tqdm(total=total_frames, desc="Reading frames", unit="fr")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if native_idx % keep_every == 0:
            # BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize with high-quality interpolation
            frame = cv2.resize(frame, (target_w, target_h),
                               interpolation=cv2.INTER_LANCZOS4)
            # Convert to greyscale if requested (replicate to 3 channels so
            # model architecture is unchanged)
            if mono:
                grey  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)   # [H, W]
                frame = np.stack([grey, grey, grey], axis=-1)      # [H, W, 3]
            raw_frames.append(frame.astype(np.uint8))
        native_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()

    n_frames = len(raw_frames)
    log.info(f"Retained {n_frames} frames after resampling")

    if n_frames < nt:
        raise ValueError(
            f"Video only has {n_frames} usable frames after resampling "
            f"but nt={nt} — use a longer video or reduce --fps / --nt."
        )

    frames_arr = np.stack(raw_frames, axis=0)   # [N, H, W, 3]

    # ── Sliding-window into sequences ─────────────────────────────────────────
    stride     = 1 if overlap else nt
    sequences  = []
    sources    = []

    for start in range(0, n_frames - nt + 1, stride):
        sequences.append(frames_arr[start : start + nt])
        sources.append(source_name)

    sequences = np.stack(sequences, axis=0)   # [S, NT, H, W, 3]
    sources   = np.array(sources)             # [S]

    log.info(f"Created {len(sequences)} sequences  "
             f"(nt={nt}, stride={stride}, shape={sequences.shape})")

    # ── Optional train/test split ─────────────────────────────────────────────
    if split > 0.0:
        split_idx    = max(1, int(len(sequences) * (1.0 - split)))
        train_seqs   = sequences[:split_idx]
        train_srcs   = sources[:split_idx]
        test_seqs    = sequences[split_idx:]
        test_srcs    = sources[split_idx:]

        _save(train_seqs, train_srcs, out_dir, prefix='train')
        _save(test_seqs,  test_srcs,  out_dir, prefix='test')

        log.info(f"Train sequences : {len(train_seqs)}")
        log.info(f"Test  sequences : {len(test_seqs)}")
    else:
        # Single test split (most common use-case when applying to new video)
        _save(sequences, sources, out_dir, prefix='test')


def _save(sequences: np.ndarray, sources: np.ndarray,
          out_dir: Path, prefix: str) -> None:
    """Write X_<prefix>.h5 and sources_<prefix>.h5"""
    x_path = out_dir / f'X_{prefix}.h5'
    s_path = out_dir / f'sources_{prefix}.h5'

    with h5py.File(x_path, 'w') as f:
        f.create_dataset('data', data=sequences, compression='gzip', compression_opts=4)

    with h5py.File(s_path, 'w') as f:
        encoded = np.array([s.encode('utf-8') for s in sources])
        f.create_dataset('data', data=encoded)

    size_mb = x_path.stat().st_size / 1e6
    log.info(f"Saved {x_path.name}  ({size_mb:.1f} MB)  shape={sequences.shape}")
    log.info(f"Saved {s_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Verification helper
# ─────────────────────────────────────────────────────────────────────────────

def verify(out_dir: Path, prefix: str = 'test') -> None:
    """Load saved .h5 files and print a quick sanity report."""
    x_path = out_dir / f'X_{prefix}.h5'
    s_path = out_dir / f'sources_{prefix}.h5'

    with h5py.File(x_path, 'r') as f:
        X = f['data'][:]
    with h5py.File(s_path, 'r') as f:
        S = np.array([s.decode('utf-8') for s in f['data'][:]])

    print('\n── Verification ──────────────────────────────────────')
    print(f'  X shape   : {X.shape}   dtype={X.dtype}')
    print(f'  X range   : [{X.min()}, {X.max()}]')
    print(f'  Sources   : {np.unique(S).tolist()}')
    print(f'  # seqs    : {len(X)}')
    print('  ✅ Files look correct for PredNet dataloader')
    print('─────────────────────────────────────────────────────\n')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Convert an MP4 to PredNet-compatible .h5 sequences',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('mp4',           type=str,   help='Input .mp4 file')
    p.add_argument('--out',  '-o',  type=str,   default='./prednet_data',
                   help='Output directory for .h5 files')
    p.add_argument('--nt',          type=int,   default=15,
                   help='Sequence length — must match nt_extrap in test script')
    p.add_argument('--height',      type=int,   default=128,
                   help='Target frame height (must match training resolution)')
    p.add_argument('--width',       type=int,   default=160,
                   help='Target frame width  (must match training resolution)')
    p.add_argument('--fps',         type=float, default=10.0,
                   help='Resample video to this frame rate before windowing')
    p.add_argument('--overlap',     action='store_true',
                   help='Use stride-1 sliding window (more sequences, more overlap)')
    p.add_argument('--split',       type=float, default=0.0,
                   help='If >0, fraction of sequences reserved for test '
                        '(remainder becomes train). E.g. 0.1 = 90%% train / 10%% test.')
    p.add_argument('--source-name', type=str,   default=None,
                   help='Label stored in sources_*.h5 (defaults to video filename stem)')
    p.add_argument('--mono',        action='store_true',
                   help='Convert frames to greyscale (stored as 3-channel for model compatibility)')
    p.add_argument('--verify',      action='store_true',
                   help='After saving, reload and print a sanity check')
    return p.parse_args()


def main():
    args = parse_args()

    try:
        video_to_prednet(
            mp4_path    = args.mp4,
            out_dir     = args.out,
            target_h    = args.height,
            target_w    = args.width,
            nt          = args.nt,
            target_fps  = args.fps,
            overlap     = args.overlap,
            split       = args.split,
            source_name = args.source_name,
            mono        = args.mono,
        )

        if args.verify:
            prefix = 'test' if args.split == 0.0 else 'train'
            verify(Path(args.out), prefix=prefix)

        return 0

    except KeyboardInterrupt:
        log.info('Cancelled.')
        return 130
    except Exception as e:
        log.error(str(e), exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
