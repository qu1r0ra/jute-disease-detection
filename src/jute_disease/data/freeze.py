"""The freeze workflow behind ``jute-disease freeze``.

Three steps share one output directory:

* :func:`freeze_target` fingerprints the target images, applies the curation
  rules and identity policy, and writes the manifest with its receipt last.
* :func:`probe_cache` builds the bounded train/validation image cache and records
  the cache decision from measured time and disk use.
* :func:`intake_source` records a pinned source package, derives its grouped
  validation split and quarantines source images that match the frozen target
  validation or test images.

None of them fits a model or reads test pixels for any purpose but fingerprinting.
"""

import json
from pathlib import Path

from jute_disease.config import (
    CURATION_RULES_PATH,
    DEFAULT_SEED,
    IDENTITY_REVIEWS_PATH,
    SOURCE_INVENTORY_DIR,
)
from jute_disease.data.cache import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_MAX_BYTES,
    build_image_cache,
    cache_root,
    decide_cache,
    write_decision,
)
from jute_disease.data.curation import load_rules
from jute_disease.data.identity import load_reviews
from jute_disease.data.inventory import (
    build_inventory,
    read_inventory,
    write_inventory,
)
from jute_disease.data.manifest import (
    FreezeBlockedError,
    build_manifest,
    load_manifest,
    receipt_for,
    write_manifest,
)
from jute_disease.data.sources import (
    PLANTDOC,
    PLANTVILLAGE,
    SourceSpec,
    check_overlap,
    derive_validation,
    drop_overlap,
    leaf_constraints,
    provenance,
    scan_source,
)
from jute_disease.data.transforms import PreprocessingSpec

TARGET_DIR = "target"
INVENTORY_FILE = "inventory.jsonl"
REPORT_FILE = "freeze_report.json"
SOURCES = {"plantdoc": PLANTDOC, "plantvillage": PLANTVILLAGE}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def freeze_target(
    output_dir: Path,
    image_root: Path = SOURCE_INVENTORY_DIR,
    rules_path: Path = CURATION_RULES_PATH,
    reviews_path: Path | None = IDENTITY_REVIEWS_PATH,
    seed: int = DEFAULT_SEED,
    workers: int = 4,
    force: bool = False,
) -> dict:
    """Freeze the target dataset and return its report.

    The original images under ``image_root`` are only read. An existing completed
    freeze that differs from the new one is kept unless ``force`` is set.

    Raises:
        FreezeBlockedError: the dataset cannot be frozen as specified, or a different
            completed freeze already exists.
    """
    output_dir = Path(output_dir)
    rules = load_rules(rules_path)
    reviews = load_reviews(reviews_path)
    records = build_inventory(image_root, workers=workers)
    manifest, curation, resolution = build_manifest(
        records,
        rules,
        reviews,
        seed=seed,
        extra_dependencies={"preprocessing": PreprocessingSpec().digest()},
    )
    target = output_dir / TARGET_DIR
    try:
        existing = load_manifest(target)
    except (FileNotFoundError, FreezeBlockedError):
        existing = None
    if existing is not None and existing.digest() != manifest.digest() and not force:
        raise FreezeBlockedError(
            f"A different completed freeze exists in {target}; "
            "inspect it, then rerun with --force to replace it"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_inventory(records, output_dir / INVENTORY_FILE)
    write_manifest(manifest, target)
    report = {
        "image_root": str(image_root),
        "receipt": receipt_for(manifest),
        "curation": curation.summary(),
        "identity": resolution.summary(),
        "unchanged": existing is not None and existing.digest() == manifest.digest(),
    }
    _write_json(output_dir / REPORT_FILE, report)
    return report


def probe_cache(
    output_dir: Path,
    image_root: Path = SOURCE_INVENTORY_DIR,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    workers: int = 4,
) -> dict:
    """Build the bounded train/validation cache and record the cache decision."""
    output_dir = Path(output_dir)
    manifest = load_manifest(output_dir / TARGET_DIR)
    spec = PreprocessingSpec()
    report = build_image_cache(
        manifest,
        image_root,
        output_dir / "image_cache",
        spec,
        budget_seconds=budget_seconds,
        workers=workers,
    )
    decision = decide_cache(report, max_bytes=max_bytes)
    root = cache_root(output_dir / "image_cache", spec)
    write_decision(decision, root)
    return decision


def intake_source(
    name: str,
    source_root: Path,
    output_dir: Path,
    package_version: int | None = None,
    leaf_map_path: Path | None = None,
    seed: int = DEFAULT_SEED,
    workers: int = 4,
) -> dict:
    """Record a source package and check it against the frozen target.

    Raises:
        KeyError: ``name`` is not a known source.
        SourceMismatchError: the package does not match its pinned identity.
        FileNotFoundError: the target has not been frozen yet.
    """
    spec: SourceSpec = SOURCES[name.lower()]
    output_dir = Path(output_dir)
    manifest = load_manifest(output_dir / TARGET_DIR)
    held_out = {
        e.image_id for name_ in ("val", "test") for e in manifest.partition(name_)
    }
    target = [
        r for r in read_inventory(output_dir / INVENTORY_FILE) if r.image_id in held_out
    ]

    records = scan_source(source_root, spec, workers=workers)
    leaf_map = (
        None
        if leaf_map_path is None
        else json.loads(Path(leaf_map_path).read_text(encoding="utf-8"))
    )
    train_records = [r for r in records if r.source_partition == "train"]
    extra = () if leaf_map is None else leaf_constraints(train_records, leaf_map)
    split = derive_validation(records, seed=seed, extra_constraints=extra)
    overlap = check_overlap(records, target)
    split, removed = drop_overlap(split, overlap)

    result = provenance(spec, records, package_version, split, overlap, leaf_map)
    result["removed"] = [r.path for r in removed]
    _write_json(output_dir / f"source_{spec.name.lower()}.json", result)
    return result
