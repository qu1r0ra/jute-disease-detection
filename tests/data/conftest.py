"""Shared synthetic fixtures for the data-layer tests (no real data is touched)."""

from pathlib import Path

import pytest

from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data.curation import CurationRules, load_rules
from jute_disease.data.inventory import InventoryRecord, build_inventory
from jute_disease.data.manifest import Manifest, build_manifest
from jute_disease.data.synthetic import SyntheticPlan, make_synthetic_data


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> SyntheticPlan:
    root = tmp_path_factory.mktemp("synthetic") / "tree"
    return make_synthetic_data(root, per_class=20, size=64, quarantine_count=4)


@pytest.fixture(scope="session")
def inventory(synthetic: SyntheticPlan) -> list[InventoryRecord]:
    return build_inventory(synthetic.root, workers=2)


@pytest.fixture(scope="session")
def rules() -> CurationRules:
    loaded = load_rules(CURATION_RULES_PATH)
    return CurationRules(
        supported_labels=loaded.supported_labels,
        excluded_labels=loaded.excluded_labels,
        quarantine=loaded.quarantine,
        sources=loaded.sources,
        min_support=2,
    )


@pytest.fixture(scope="session")
def frozen(inventory: list[InventoryRecord], rules: CurationRules) -> Manifest:
    return build_manifest(inventory, rules, seed=42)[0]


def build_tree(root: Path, **kwargs: object) -> SyntheticPlan:
    return make_synthetic_data(root, **kwargs)  # type: ignore[arg-type]
