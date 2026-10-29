"""The shared freeze command, run as a user would, on the synthetic fixture."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data.inventory import inventory_digest, read_inventory
from jute_disease.data.manifest import load_manifest
from jute_disease.data.synthetic import make_synthetic_data


pytestmark = pytest.mark.slow


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "jute_disease.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("freeze-command")
    make_synthetic_data(base / "images", per_class=20, quarantine_count=4)
    rules = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    rules["min_support"] = 2
    (base / "rules.json").write_text(json.dumps(rules), encoding="utf-8")
    (base / "reviews.json").write_text(
        json.dumps({"version": 1, "decisions": []}), encoding="utf-8"
    )
    return {
        "images": base / "images",
        "rules": base / "rules.json",
        "reviews": base / "reviews.json",
        "base": base,
    }


def _freeze(inputs: dict[str, Path], name: str, *extra: str):
    out = inputs["base"] / name
    result = _run(
        "freeze",
        "target",
        "--image-root",
        str(inputs["images"]),
        "--rules",
        str(inputs["rules"]),
        "--reviews",
        str(inputs["reviews"]),
        "--output",
        str(out),
        "--workers",
        "2",
        *extra,
    )
    return out, result


def test_freeze_command_writes_a_verifiable_receipt(inputs: dict[str, Path]) -> None:
    out, result = _freeze(inputs, "first")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    manifest = load_manifest(out / "target")
    receipt = report["receipt"]
    assert receipt["manifest_digest"] == manifest.digest()
    assert receipt["seed"] == 42
    assert receipt["fractions"] == {"train": 0.7, "val": 0.15, "test": 0.15}
    dependencies = receipt["dependencies"]
    assert {"inventory", "rules", "reviews", "identity", "preprocessing"} <= set(
        dependencies
    )
    assert dependencies["inventory"] == inventory_digest(
        read_inventory(out / "inventory.jsonl")
    )


def test_lineage_a_second_process_reproduces_the_same_freeze(
    inputs: dict[str, Path],
) -> None:
    first_out, first = _freeze(inputs, "lineage-a")
    second_out, second = _freeze(inputs, "lineage-b")
    assert first.returncode == second.returncode == 0
    assert json.loads(first.stdout)["receipt"] == json.loads(second.stdout)["receipt"]
    for name in ("manifest.json", "receipt.json"):
        assert (first_out / "target" / name).read_bytes() == (
            second_out / "target" / name
        ).read_bytes()


def test_regression_a_different_freeze_is_not_replaced_without_force(
    inputs: dict[str, Path],
) -> None:
    out, first = _freeze(inputs, "guarded")
    assert first.returncode == 0
    before = (out / "target" / "receipt.json").read_bytes()
    _, refused = _freeze(inputs, "guarded", "--seed", "7")
    assert refused.returncode != 0
    assert "--force" in refused.stderr
    assert (out / "target" / "receipt.json").read_bytes() == before
    _, forced = _freeze(inputs, "guarded", "--seed", "7", "--force")
    assert forced.returncode == 0
    assert (out / "target" / "receipt.json").read_bytes() != before


def test_regression_a_changed_input_changes_its_dependency_hash(
    inputs: dict[str, Path],
) -> None:
    _, base = _freeze(inputs, "dep-base")
    rules = json.loads(inputs["rules"].read_text(encoding="utf-8"))
    rules["min_support"] = 3
    changed = inputs["base"] / "rules-changed.json"
    changed.write_text(json.dumps(rules), encoding="utf-8")
    other = {**inputs, "rules": changed}
    _, moved = _freeze(other, "dep-moved")
    old = json.loads(base.stdout)["receipt"]["dependencies"]
    new = json.loads(moved.stdout)["receipt"]["dependencies"]
    assert old["rules"] != new["rules"]
    assert old["inventory"] == new["inventory"]
