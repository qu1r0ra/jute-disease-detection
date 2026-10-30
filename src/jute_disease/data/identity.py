"""Identity resolution: exact duplicates, review decisions and canonicalization.

Policy:

* Byte-identical or pixel-identical files are the same image. Identical files with
  different labels are quarantined.
* A perceptual match within the threshold is only a *candidate*. A reviewer may
  confirm it (``same``) or resolve it (``distinct``).
* Confirmed images collapse to one canonical file: the reviewer-named original,
  else the largest pixel area, else the smallest byte hash. The others are merged
  away and recorded, never deleted.
* Unresolved candidates are never silently ignored. Same-label ones become
  same-partition constraints; cross-label ones quarantine both files.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jute_disease.data.fingerprints import PHASH_THRESHOLD
from jute_disease.data.inventory import InventoryRecord

REVIEW_VERSION = 1
_DECISIONS = {"same", "distinct"}
_BLOCK = 256


@dataclass(frozen=True)
class Candidate:
    """A perceptual match between two retained images (``a < b`` by image id)."""

    a: str
    b: str
    distance: int
    same_label: bool


@dataclass(frozen=True)
class Review:
    """A reviewer decision about one pair of images."""

    ids: tuple[str, str]
    decision: str
    original: str | None
    evidence: str


@dataclass(frozen=True)
class MergedFile:
    """A file collapsed into a canonical image."""

    path: str
    image_id: str
    canonical_id: str
    reason: str


@dataclass(frozen=True)
class QuarantinedFile:
    """A file withheld from every partition."""

    path: str
    image_id: str
    reason: str


@dataclass(frozen=True)
class IdentityResolution:
    """Outcome of identity resolution over an admitted inventory."""

    retained: tuple[InventoryRecord, ...]
    merged: tuple[MergedFile, ...]
    quarantined: tuple[QuarantinedFile, ...]
    constraints: tuple[tuple[str, str], ...]
    candidates: tuple[tuple[Candidate, str], ...]

    def summary(self) -> dict:
        """Return counts describing the resolution."""
        statuses: dict[str, int] = {}
        for _, status in self.candidates:
            statuses[status] = statuses.get(status, 0) + 1
        return {
            "retained": len(self.retained),
            "merged": len(self.merged),
            "quarantined": len(self.quarantined),
            "constraints": len(self.constraints),
            "candidates": len(self.candidates),
            "candidate_status": dict(sorted(statuses.items())),
            "cross_label_candidates": sum(
                1 for c, _ in self.candidates if not c.same_label
            ),
        }


def load_reviews(path: Path | None) -> tuple[Review, ...]:
    """Load reviewer decisions; a missing path means no decisions yet.

    Raises:
        ValueError: the file is malformed or contradicts itself.
    """
    if path is None:
        return ()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != REVIEW_VERSION:
        raise ValueError(f"Unsupported review file version: {data.get('version')!r}")
    reviews: dict[tuple[str, str], Review] = {}
    for item in data.get("decisions", []):
        ids = tuple(sorted(item["ids"]))
        if len(ids) != 2 or ids[0] == ids[1]:
            raise ValueError(f"A review needs two distinct image ids: {item!r}")
        if item["decision"] not in _DECISIONS:
            raise ValueError(f"Unknown review decision: {item['decision']!r}")
        original = item.get("original")
        if original is not None and original not in ids:
            raise ValueError(f"Review original is not one of its ids: {item!r}")
        if original is not None and item["decision"] != "same":
            raise ValueError(f"Only 'same' reviews can name an original: {item!r}")
        review = Review(ids, item["decision"], original, item.get("evidence", ""))
        previous = reviews.get(ids)
        if previous is not None and previous != review:
            raise ValueError(f"Conflicting reviews for pair {ids}")
        reviews[ids] = review
    return tuple(reviews[key] for key in sorted(reviews))


def reviews_digest(reviews: tuple[Review, ...]) -> str:
    """Return a digest of the reviewer decisions, independent of file formatting."""
    payload = json.dumps(
        [
            [list(r.ids), r.decision, r.original, r.evidence]
            for r in sorted(reviews, key=lambda r: r.ids)
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def find_candidates(
    records: list[InventoryRecord], threshold: int = PHASH_THRESHOLD
) -> list[Candidate]:
    """Return every pair whose perceptual distance is within ``threshold``.

    The distance is symmetric: the smaller of the two directional dihedral
    distances, so the result does not depend on record order. Records must have
    distinct image ids.
    """
    ordered = sorted(records, key=lambda r: r.image_id)
    if len({r.image_id for r in ordered}) != len(ordered):
        raise ValueError("find_candidates needs records with distinct image ids")
    if len(ordered) < 2:
        return []
    variants = np.array([r.phash for r in ordered], dtype=np.uint64)
    base = variants[:, 0]
    found: list[Candidate] = []
    for start in range(0, len(ordered), _BLOCK):
        rows = slice(start, min(start + _BLOCK, len(ordered)))
        forward = np.bitwise_count(base[rows, None, None] ^ variants[None]).min(axis=2)
        backward = np.bitwise_count(variants[rows, None, :] ^ base[None, :, None]).min(
            axis=2
        )
        distance = np.minimum(forward, backward)
        for local, j in zip(*np.nonzero(distance <= threshold), strict=True):
            i = start + int(local)
            if i < j:
                found.append(
                    Candidate(
                        a=ordered[i].image_id,
                        b=ordered[j].image_id,
                        distance=int(distance[local, j]),
                        same_label=ordered[i].label == ordered[j].label,
                    )
                )
    return sorted(found, key=lambda c: (c.a, c.b))


class _Clusters:
    """Union-find over string keys with deterministic roots."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _canonical(members: list[InventoryRecord], originals: set[str]) -> InventoryRecord:
    named = [m for m in members if m.image_id in originals]
    pool = named or members
    return sorted(pool, key=lambda m: (0 if named else -m.area, m.image_id, m.path))[0]


def resolve_identity(
    records: list[InventoryRecord],
    reviews: tuple[Review, ...] = (),
    threshold: int = PHASH_THRESHOLD,
) -> IdentityResolution:
    """Resolve exact identity, reviewed identity and unresolved candidates.

    Raises:
        ValueError: a review names an image that is not in ``records``.
    """
    known = {r.image_id for r in records}
    for review in reviews:
        for image_id in review.ids:
            if image_id not in known:
                raise ValueError(f"Review references an unknown image: {image_id}")

    by_id: dict[str, list[InventoryRecord]] = {}
    for record in sorted(records, key=lambda r: (r.image_id, r.path)):
        by_id.setdefault(record.image_id, []).append(record)

    quarantined: dict[str, QuarantinedFile] = {}
    merged: dict[str, MergedFile] = {}
    unique: dict[str, InventoryRecord] = {}
    for image_id, files in by_id.items():
        representative = files[0]
        unique[image_id] = representative
        if len({f.label for f in files}) > 1:
            for f in files:
                quarantined[f.path] = QuarantinedFile(
                    f.path, image_id, "byte-identical files with conflicting labels"
                )
            continue
        for extra in files[1:]:
            merged[extra.path] = MergedFile(
                extra.path, image_id, image_id, "byte-identical copy"
            )

    clusters = _Clusters()
    pixel_groups: dict[str, list[str]] = {}
    for image_id, record in unique.items():
        clusters.find(image_id)
        pixel_groups.setdefault(record.pixel_sha256, []).append(image_id)
    for group in pixel_groups.values():
        for other in group[1:]:
            clusters.union(group[0], other)
    originals = {r.original for r in reviews if r.original}
    for review in reviews:
        if review.decision == "same":
            clusters.union(*review.ids)

    members: dict[str, list[InventoryRecord]] = {}
    for image_id, record in unique.items():
        members.setdefault(clusters.find(image_id), []).append(record)
    canonical_of: dict[str, str] = {}
    cluster_of: dict[str, str] = {}
    for root, group in members.items():
        chosen = _canonical(group, originals)
        cluster_labels = {m.label for m in group}
        for m in group:
            canonical_of[m.image_id] = chosen.image_id
            cluster_of[m.image_id] = root
            if len(cluster_labels) > 1:
                for f in by_id[m.image_id]:
                    quarantined[f.path] = QuarantinedFile(
                        f.path, m.image_id, "confirmed identity across labels"
                    )
            elif m.image_id != chosen.image_id:
                for f in by_id[m.image_id]:
                    reason = (
                        "identical decoded pixels"
                        if m.pixel_sha256 == chosen.pixel_sha256
                        else "reviewed as the same image"
                    )
                    merged[f.path] = MergedFile(
                        f.path, m.image_id, chosen.image_id, reason
                    )

    decided = {r.ids: r.decision for r in reviews}
    candidates: list[tuple[Candidate, str]] = []
    constraints: set[tuple[str, str]] = set()
    pair_quarantine: set[str] = set()
    for candidate in find_candidates(list(unique.values()), threshold):
        decision = decided.get((candidate.a, candidate.b))
        same_cluster = cluster_of[candidate.a] == cluster_of[candidate.b]
        if same_cluster:
            candidates.append((candidate, "confirmed"))
        elif decision == "distinct":
            candidates.append((candidate, "distinct"))
        elif candidate.same_label:
            candidates.append((candidate, "unresolved-same-label"))
            pair = tuple(sorted((canonical_of[candidate.a], canonical_of[candidate.b])))
            constraints.add(pair)
        else:
            candidates.append((candidate, "unresolved-cross-label"))
            pair_quarantine.update((candidate.a, candidate.b))
    for image_id in sorted(pair_quarantine):
        for member_id, root in cluster_of.items():
            if root == cluster_of[image_id]:
                for f in by_id[member_id]:
                    quarantined.setdefault(
                        f.path,
                        QuarantinedFile(
                            f.path, member_id, "unresolved cross-label candidate"
                        ),
                    )

    withheld = {q.image_id for q in quarantined.values()}
    retained = tuple(
        sorted(
            (
                unique[image_id]
                for image_id in set(canonical_of.values())
                if image_id not in withheld
            ),
            key=lambda r: r.image_id,
        )
    )
    kept = {r.image_id for r in retained}
    return IdentityResolution(
        retained=retained,
        merged=tuple(sorted(merged.values(), key=lambda m: m.path)),
        quarantined=tuple(sorted(quarantined.values(), key=lambda q: q.path)),
        constraints=tuple(
            sorted(
                c for c in constraints if c[0] in kept and c[1] in kept and c[0] != c[1]
            )
        ),
        candidates=tuple(candidates),
    )
