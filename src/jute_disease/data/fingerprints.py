"""Content fingerprints used for identity checks and leakage control.

Three signals are computed per image:

* ``sha256``: hash of the file bytes (exact byte identity).
* ``pixel_sha256``: hash of the decoded RGB size and pixels (exact pixel identity).
* ``phash``: eight 64-bit DCT perceptual hashes, one per dihedral variant
  (four rotations of the image and of its mirror image).

Perceptual matches are review candidates, never proof of identity.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

PHASH_THRESHOLD = 6
PHASH_VARIANTS = 8
_HASH_GRID = 32
_HASH_BLOCK = 8
_CHUNK = 1 << 20


@dataclass(frozen=True)
class ImageFingerprint:
    """Content signals for one image file."""

    bytes: int
    width: int
    height: int
    sha256: str
    pixel_sha256: str
    phash: tuple[int, ...]


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hashes(rgb: Image.Image) -> tuple[int, ...]:
    """Return the dihedral 64-bit DCT hashes of an RGB image.

    The image is reduced to a 32x32 grayscale grid, the 8x8 low-frequency block
    of its 2-D DCT is thresholded at the median with the DC term zeroed, and the
    result is repeated for the eight rotations/reflections of the grid. The first
    entry is the hash of the unmodified image.
    """
    gray = np.asarray(
        rgb.convert("L").resize((_HASH_GRID, _HASH_GRID), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    hashes: list[int] = []
    for flip in (False, True):
        base = np.fliplr(gray) if flip else gray
        for turns in range(4):
            block = dctn(np.rot90(base, turns), type=2, norm="ortho")[
                :_HASH_BLOCK, :_HASH_BLOCK
            ].copy()
            block[0, 0] = 0
            bits = (block > np.median(block)).ravel()
            hashes.append(int.from_bytes(np.packbits(bits).tobytes(), "big"))
    return tuple(hashes)


def phash_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """Return the smallest Hamming distance between ``left``'s base hash and any
    dihedral variant of ``right``."""
    return min((left[0] ^ variant).bit_count() for variant in right)


def fingerprint_image(path: Path) -> ImageFingerprint:
    """Compute all content signals for one image file."""
    path = Path(path)
    sha256 = file_sha256(path)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    width, height = rgb.size
    pixel_sha256 = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    return ImageFingerprint(
        bytes=path.stat().st_size,
        width=width,
        height=height,
        sha256=sha256,
        pixel_sha256=pixel_sha256,
        phash=perceptual_hashes(rgb),
    )
