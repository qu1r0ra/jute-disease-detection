"""Source-preserving curation rules.

Curation never edits or removes files. It decides which inventoried files are
*admitted* to the study, and records every other file with a reason:

* unsupported labels (not among the study classes),
* explicitly excluded labels,
* quarantined files (a filename pattern within a label, e.g. a contribution whose
  labels could not be substantiated).

It also tags each file with its provisional source contribution.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from jute_disease.data.inventory import InventoryRecord

RULES_VERSION = 1
UNASSIGNED_SOURCE = "unassigned"


@dataclass(frozen=True)
class QuarantineRule:
    """Withhold files of ``label`` whose filename matches ``filename_regex``."""

    label: str
    filename_regex: str
    source: str
    reason: str


@dataclass(frozen=True)
class SourceRule:
    """Tag files with a provisional source contribution."""

    source: str
    filename_regex: str
    label: str | None = None


@dataclass(frozen=True)
class CurationRules:
    """Declarative curation rules; hashed as a dependency of every freeze."""

    supported_labels: tuple[str, ...]
    excluded_labels: tuple[tuple[str, str], ...]
    quarantine: tuple[QuarantineRule, ...]
    sources: tuple[SourceRule, ...]
    min_support: int

    def to_json(self) -> dict:
        """Return a canonical JSON-safe mapping."""
        return {
            "version": RULES_VERSION,
            "supported_labels": sorted(self.supported_labels),
            "excluded_labels": dict(self.excluded_labels),
            "quarantine": [vars(rule) for rule in self.quarantine],
            "sources": [vars(rule) for rule in self.sources],
            "min_support": self.min_support,
        }

    def digest(self) -> str:
        """Return the SHA-256 of the canonical rules."""
        payload = json.dumps(self.to_json(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ExcludedFile:
    """An inventoried file withheld from the study."""

    path: str
    image_id: str
    label: str
    kind: str
    reason: str
    source: str


@dataclass(frozen=True)
class Curation:
    """Outcome of applying rules to an inventory."""

    admitted: tuple[InventoryRecord, ...]
    excluded: tuple[ExcludedFile, ...]
    sources: dict[str, str]

    def summary(self) -> dict:
        """Return per-label and per-source counts for provenance records."""
        admitted: dict[str, int] = {}
        for record in self.admitted:
            admitted[record.label] = admitted.get(record.label, 0) + 1
        excluded: dict[str, int] = {}
        for item in self.excluded:
            key = f"{item.kind}:{item.label}"
            excluded[key] = excluded.get(key, 0) + 1
        by_source: dict[str, int] = {}
        for source in self.sources.values():
            by_source[source] = by_source.get(source, 0) + 1
        return {
            "admitted": len(self.admitted),
            "excluded": len(self.excluded),
            "admitted_by_label": dict(sorted(admitted.items())),
            "excluded_by_kind_and_label": dict(sorted(excluded.items())),
            "files_by_source": dict(sorted(by_source.items())),
        }


def load_rules(path: Path) -> CurationRules:
    """Load curation rules from JSON.

    Raises:
        ValueError: the version is unsupported or the rules are inconsistent.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != RULES_VERSION:
        raise ValueError(f"Unsupported rules version: {data.get('version')!r}")
    supported = tuple(sorted(data["supported_labels"]))
    excluded = tuple(sorted(data.get("excluded_labels", {}).items()))
    if not supported:
        raise ValueError("Curation rules must list at least one supported label")
    overlap = set(supported) & {label for label, _ in excluded}
    if overlap:
        raise ValueError(f"Labels are both supported and excluded: {sorted(overlap)}")
    quarantine = tuple(QuarantineRule(**rule) for rule in data.get("quarantine", []))
    sources = tuple(SourceRule(**rule) for rule in data.get("sources", []))
    for rule in (*quarantine, *sources):
        re.compile(rule.filename_regex)
    return CurationRules(
        supported_labels=supported,
        excluded_labels=excluded,
        quarantine=quarantine,
        sources=sources,
        min_support=int(data.get("min_support", 1)),
    )


def source_of(record: InventoryRecord, rules: CurationRules) -> str:
    """Return the provisional source tag of a file (first matching rule)."""
    name = Path(record.path).name
    for rule in rules.sources:
        if rule.label is not None and rule.label != record.label:
            continue
        if re.search(rule.filename_regex, name):
            return rule.source
    return UNASSIGNED_SOURCE


def curate(records: list[InventoryRecord], rules: CurationRules) -> Curation:
    """Split an inventory into admitted and excluded files.

    Raises:
        ValueError: a supported label has no admitted file, which would silently
            drop a class from the study.
    """
    excluded_labels = dict(rules.excluded_labels)
    admitted: list[InventoryRecord] = []
    excluded: list[ExcludedFile] = []
    sources: dict[str, str] = {}
    for record in sorted(records, key=lambda r: r.path):
        source = source_of(record, rules)
        sources[record.path] = source
        reason: tuple[str, str] | None = None
        if record.label in excluded_labels:
            reason = ("excluded-label", excluded_labels[record.label])
        elif record.label not in rules.supported_labels:
            reason = ("unsupported-label", "label is not a study class")
        else:
            for rule in rules.quarantine:
                if rule.label == record.label and re.search(
                    rule.filename_regex, Path(record.path).name
                ):
                    reason = ("quarantine", rule.reason)
                    source = rule.source
                    sources[record.path] = source
                    break
        if reason is None:
            admitted.append(record)
        else:
            excluded.append(
                ExcludedFile(
                    record.path,
                    record.image_id,
                    record.label,
                    reason[0],
                    reason[1],
                    source,
                )
            )
    present = {record.label for record in admitted}
    missing = sorted(set(rules.supported_labels) - present)
    if missing:
        raise ValueError(f"Supported labels have no admitted images: {missing}")
    return Curation(tuple(admitted), tuple(excluded), sources)
