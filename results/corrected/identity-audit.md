# Target identity audit receipt

The recovered historical inventory contains 2,382 files (train 1,666, val 356,
test 360). The preserved audit found 1,104 candidate pairs, including 465 across
historical partitions and 39 across labels. These are heuristic candidates, not
confirmed photographs. The historical script sorted paths, compared the base hash
of the earlier path to each of eight hashes of the later path, and retained the
minimum Hamming distance at most six. It decoded RGB, resized grayscale to 32x32
with Lanczos, used an orthonormal type-2 DCT, zeroed DC, and thresholded the 8x8
low-frequency block against its median. Rotations and reflections operate on the
gray grid. The archived script SHA-256 is
`865cd2cfbbc5c6df3c542b2473a79a7c02329524295f34af300e3cf310491569`;
the audit summary is `b90a5808ab0e3fe6eb7006088970eb252a2f60759911c9b3f95f33fa69c80f8e`;
the original fingerprint inventory is
`944609d70c91841bf4eba413921080484904b1b042ea8ccff000c7e21c8a7f24`.

The current algorithm computes the minimum of both directional comparisons.
Recomputing against the preserved full inventory produced 1,144 pairs, 488
crossing historical splits and 50 crossing labels. All 1,104 historical pairs
were present, with 40 additional pairs. The current freeze first excludes 72
unsupported Healthy files; on the remaining 2,310 files it reported 1,006 pairs,
including 50 cross-label candidates. Thus the counts use different populations and
distance direction rules; neither is a corrected historical split or label.

The existing freeze reported 60 quarantined cross-label files, zero merged files,
2,250 retained files and 922 retained constraints. Its 1,690 components have
sizes: 1,180 singletons, 503 pairs, two triples, three of four, one of 18 and
one of 28. Large components reflect transitive constraints, not confirmed
identity; chains may link endpoints that do not match directly. Six-class support
(train/val/test) was Cercospora 207/44/44, Dieback 210/45/45, General Damage
210/45/45, Healthy 361/78/77, Mosaic 398/86/85 and Stem Rot 189/41/40.
Manifest SHA-256: `300769862d3e3c7c11e176afed63f8c2a0f38ed873d71c20e3b93d74c22ed25f`.

The archived sample contact sheets and alignment comparisons do not establish
individual adjudication decisions or a timed 60-minute review. The review file
contains no decisions; elapsed human review time is unavailable. Consequently,
unresolved same-label candidates retain distinct IDs with shared-partition
constraints, and unresolved cross-label candidates are quarantined. Neither the
sample inspection nor these fingerprints establishes expert label validation or
exhaustive absence of leakage. Reproduce the full comparison using the preserved
historical `audit-summary.json`, the freeze `inventory.jsonl`, and
`find_candidates`: compare unordered relative-path pairs, then filter the current
inventory using the recorded curation rules. The source artifacts remain private
and are not redistributed here.
