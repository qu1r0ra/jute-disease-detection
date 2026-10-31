# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: jute-related-image-evaluation (3.11.14)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Freeze Provenance
#
# This notebook shows how a frozen dataset is produced, checked and traced. It runs
# the shared `jute-disease freeze target` command on a synthetic fixture, so it needs
# no real images and fits no model.
#
# - **Demonstration:** one command writes the manifest and its receipt.
# - **Lineage:** every input is hashed into the receipt; a second process reproduces it.
# - **Regression:** a changed input changes only its own hash, and a different freeze
#   is never replaced silently.

# %%
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import load_manifest, make_synthetic_data

work = Path(tempfile.mkdtemp(prefix="freeze-provenance-"))
plan = make_synthetic_data(work / "images", per_class=20, quarantine_count=4)

rules = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
rules["min_support"] = 2  # the fixture is small; the real rules require more
(work / "rules.json").write_text(json.dumps(rules), encoding="utf-8")
(work / "reviews.json").write_text(
    json.dumps({"version": 1, "decisions": []}), encoding="utf-8"
)
print(
    f"{len(plan.files)} synthetic images; {len(plan.quarantine_files)} are quarantined by rule"
)


# %%
def freeze(name: str, rules_file: str = "rules.json", *extra: str) -> dict:
    """Run the shared freeze command in a fresh process and return its report."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jute_disease.cli",
            "freeze",
            "target",
            "--image-root",
            str(work / "images"),
            "--rules",
            str(work / rules_file),
            "--reviews",
            str(work / "reviews.json"),
            "--output",
            str(work / name),
            "--workers",
            "2",
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip().splitlines()[-1])
    return json.loads(result.stdout)


# %% [markdown]
# ## 1. Demonstration
#
# The receipt is written last, so a directory without one is never a completed freeze.

# %%
first = freeze("first")
receipt = first["receipt"]
pd.Series(
    {k: receipt[k] for k in ("seed", "images", "groups", "excluded", "quarantined")}
).to_frame("value")

# %%
pd.DataFrame(receipt["support"]).T.rename_axis("partition")

# %% [markdown]
# ## 2. Lineage
#
# Each dependency is a hash of the input that shaped the manifest. A second process
# with the same inputs must produce the same receipt and byte-identical files.

# %%
pd.Series(receipt["dependencies"]).to_frame("sha256")

# %%
second = freeze("second")
same_receipt = second["receipt"] == receipt
same_bytes = all(
    (work / "first/target" / name).read_bytes()
    == (work / "second/target" / name).read_bytes()
    for name in ("manifest.json", "receipt.json")
)
assert same_receipt and same_bytes
print("receipt equal:", same_receipt, "| files byte-identical:", same_bytes)

# %% [markdown]
# ## 3. Regression
#
# Changing a rule changes the `rules` hash and leaves the inventory hash alone. A
# different freeze into the same place is refused until it is explicitly forced.

# %%
changed = {**rules, "min_support": 3}
(work / "rules_changed.json").write_text(json.dumps(changed), encoding="utf-8")
moved = freeze("moved", "rules_changed.json")["receipt"]["dependencies"]
old = receipt["dependencies"]
pd.DataFrame(
    {"before": old, "after": moved, "changed": {k: old[k] != moved[k] for k in old}}
)

# %%
try:
    freeze("first", "rules.json", "--seed", "7")
except RuntimeError as error:
    print("refused:", str(error).split(" in ")[0], "(use --force to replace)")
else:
    raise AssertionError("a different freeze replaced the completed one")

# %% [markdown]
# ## 4. Reading the frozen partitions
#
# Partitions are read only through the manifest; the groups never span partitions.

# %%
manifest = load_manifest(work / "first/target")
frame = pd.DataFrame(
    {
        "partition": [e.partition for e in manifest.entries],
        "group": [e.group for e in manifest.entries],
    }
)
leaks = frame.groupby("group")["partition"].nunique()
assert (leaks == 1).all()
print(f"{len(leaks)} groups, none spanning partitions")
