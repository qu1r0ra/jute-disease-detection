"""Freeze workflow: target freeze, replacement guard, cache probe and source intake."""

import json
from pathlib import Path

import pytest

from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data.cache import DECISION_FILE, cache_root
from jute_disease.data.freeze import (
    INVENTORY_FILE,
    REPORT_FILE,
    TARGET_DIR,
    freeze_target,
    intake_source,
    probe_cache,
)
from jute_disease.data.manifest import FreezeBlockedError, load_manifest
from jute_disease.data.sources import PLANTDOC, SourceMismatchError
from jute_disease.data.synthetic import SyntheticPlan, make_synthetic_data
from jute_disease.data.transforms import PreprocessingSpec


@pytest.fixture(scope="module")
def rules_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    data["min_support"] = 2
    path = tmp_path_factory.mktemp("rules") / "rules.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _freeze(synthetic: SyntheticPlan, rules_path: Path, out: Path, **kwargs) -> dict:
    return freeze_target(
        out,
        image_root=synthetic.root,
        rules_path=rules_path,
        reviews_path=None,
        workers=2,
        **kwargs,
    )


def test_freeze_writes_inventory_manifest_receipt_and_report(
    synthetic: SyntheticPlan, rules_path: Path, tmp_path: Path
) -> None:
    report = _freeze(synthetic, rules_path, tmp_path)
    assert (tmp_path / INVENTORY_FILE).is_file()
    assert (
        json.loads((tmp_path / REPORT_FILE).read_text())["receipt"]
        == (report["receipt"])
    )
    manifest = load_manifest(tmp_path / TARGET_DIR)
    assert report["receipt"]["manifest_digest"] == manifest.digest()
    assert report["unchanged"] is False
    assert "preprocessing" in report["receipt"]["dependencies"]
    assert report["receipt"]["dependencies"]["preprocessing"] == (
        PreprocessingSpec().digest()
    )


def test_refreezing_the_same_inputs_is_idempotent(
    synthetic: SyntheticPlan, rules_path: Path, tmp_path: Path
) -> None:
    first = _freeze(synthetic, rules_path, tmp_path)
    again = _freeze(synthetic, rules_path, tmp_path)
    assert again["unchanged"] is True
    assert again["receipt"] == first["receipt"]


def test_a_different_completed_freeze_is_kept_unless_forced(
    synthetic: SyntheticPlan, rules_path: Path, tmp_path: Path
) -> None:
    first = _freeze(synthetic, rules_path, tmp_path)
    with pytest.raises(FreezeBlockedError, match="--force"):
        _freeze(synthetic, rules_path, tmp_path, seed=7)
    kept = load_manifest(tmp_path / TARGET_DIR)
    assert kept.digest() == first["receipt"]["manifest_digest"]
    replaced = _freeze(synthetic, rules_path, tmp_path, seed=7, force=True)
    assert replaced["receipt"]["manifest_digest"] != first["receipt"]["manifest_digest"]


def test_an_unsupported_dataset_blocks_the_freeze(
    synthetic: SyntheticPlan, tmp_path: Path
) -> None:
    with pytest.raises(FreezeBlockedError):
        freeze_target(
            tmp_path,
            image_root=synthetic.root,
            reviews_path=None,
            workers=2,
        )
    assert not (tmp_path / TARGET_DIR).exists()


def test_cache_probe_records_a_decision_and_never_reads_test(
    synthetic: SyntheticPlan, rules_path: Path, tmp_path: Path
) -> None:
    _freeze(synthetic, rules_path, tmp_path)
    decision = probe_cache(tmp_path, image_root=synthetic.root, workers=2)
    assert decision["decision"] == "adopt-cache"
    root = cache_root(tmp_path / "image_cache", PreprocessingSpec())
    assert (root / DECISION_FILE).is_file()
    manifest = load_manifest(tmp_path / TARGET_DIR)
    cached = {p.stem for p in root.glob("*.png")}
    assert cached == {
        e.image_id for name in ("train", "val") for e in manifest.partition(name)
    }


def test_cache_probe_requires_a_completed_freeze(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        probe_cache(tmp_path)


def test_source_intake_requires_a_completed_freeze(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        intake_source("plantdoc", tmp_path, tmp_path)


def test_source_intake_rejects_a_package_with_other_counts(
    synthetic: SyntheticPlan, rules_path: Path, tmp_path: Path
) -> None:
    _freeze(synthetic, rules_path, tmp_path / "freeze")
    package = tmp_path / "plantdoc"
    make_synthetic_data(package / "train", per_class=6, labels=("a", "b"))
    make_synthetic_data(package / "test", per_class=4, labels=("a", "b"), seed=3)
    with pytest.raises(SourceMismatchError, match="pinned identity"):
        intake_source("plantdoc", package, tmp_path / "freeze", package_version=7)
    assert not (tmp_path / "freeze" / "source_plantdoc.json").exists()


def test_source_names_are_pinned_case_insensitively() -> None:
    with pytest.raises(KeyError):
        intake_source("nonexistent", Path("."), Path("."))
    assert PLANTDOC.name.lower() == "plantdoc"


def test_source_freeze_excludes_target_overlap_and_pins_content(
    synthetic: SyntheticPlan,
    rules_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace
    from shutil import copyfile

    from jute_disease.data.freeze import SOURCES
    from jute_disease.data.inventory import build_inventory, inventory_digest
    from jute_disease.data.sources import observed_counts

    output = tmp_path / "freeze"
    _freeze(synthetic, rules_path, output)
    target = load_manifest(output / TARGET_DIR)
    package = tmp_path / "package"
    make_synthetic_data(package / "train", per_class=20, labels=("a", "b"), seed=100)
    make_synthetic_data(package / "test", per_class=10, labels=("a", "b"), seed=200)
    for duplicate in (package / "test").rglob("*bytecopy.png"):
        duplicate.unlink()
    overlap_path = next((package / "train" / "a").glob("*.png"))
    copyfile(synthetic.root / target.partition("val")[0].path, overlap_path)
    records = build_inventory(package, workers=2)
    spec = replace(PLANTDOC, expected=tuple(sorted(observed_counts(records).items())))
    monkeypatch.setitem(SOURCES, "fixture", spec)
    digest = inventory_digest(records)
    with pytest.raises(SourceMismatchError, match="digest mismatch"):
        intake_source(
            "fixture", package, output, package_version=7, package_digest="0" * 64
        )
    report = intake_source(
        "fixture", package, output, package_version=7, package_digest=digest
    )
    frozen = load_manifest(output / "source_plantdoc")
    assert overlap_path.relative_to(package).as_posix() in report["removed"]
    assert overlap_path.relative_to(package).as_posix() not in {
        e.path for e in frozen.entries
    }
    assert dict(frozen.dependencies)["target_manifest"] == target.digest()
    assert not {item["image_id"] for item in frozen.quarantined} & {
        entry.image_id for entry in frozen.entries
    }
    from jute_disease.training.engines.dl.pretrain import source_data_module

    dm = source_data_module(
        output / "source_plantdoc", package, output / TARGET_DIR, 4, 42
    )
    observed = {
        entry.path for ds in (dm.jute_train, dm.jute_val) for entry in ds.entries
    }
    assert overlap_path.relative_to(package).as_posix() not in observed
    kept = frozen.partition("train")[0]
    (package / kept.path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source content changed"):
        source_data_module(
            output / "source_plantdoc", package, output / TARGET_DIR, 4, 42
        )
