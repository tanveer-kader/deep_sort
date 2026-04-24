# DeepSORT Modifications for Dual-Feature Appearance Matching

## Overview

Extended DeepSORT to support two appearance feature vectors per detection — a global
feature and a set of local part features — enabling distance-level fusion at the matching
stage. The original pipeline assumes one feature vector per detection. These modifications
add a parallel feature path for local part features without breaking the existing
single-feature path.

---

## Files Modified

### `detection.py`

Added optional `local_feature=None` parameter to `Detection.__init__()`. Stores the
local part feature alongside the existing global feature. Defaults to `None` —
single-feature extractors are unaffected.

### `track.py`

Added `self.local_features = []` alongside `self.features`. Grows in lockstep with the
global feature list — every matched detection appends one entry to each list
simultaneously.

### `tracker.py`

- `update()`: flushes both `features` and `local_features` lists to their respective
  galleries via `partial_fit()`
- `_initiate_track()`: stores the detection's `local_feature` at track birth so both
  galleries start from the same first observation

### `nn_matching.py`

Four additions:

1. **`local_samples` gallery** — parallel to `samples`, stores local part features per
   track with the same budget limit

2. **`local_weight` parameter** — controls how much local features influence the final
   distance:
   ```
   fused_dist = (1 - local_weight) × global_dist + local_weight × local_dist
   ```
   `0.0` = global only, `1.0` = local only

3. **`fusion_margin` parameter** — conditional fusion gate. Local features only activate
   when global distance falls within the uncertain zone:
   ```
   threshold - margin < global_dist < threshold + margin
   ```
   Prevents local features from overriding confident global decisions.
   `0.0` = local features never activate regardless of `local_weight`

4. **`part_weights` parameter + `_nn_cosine_distance_parts()`** — splits the local
   feature into equal-sized part vectors, computes independent cosine distances per part,
   and combines with configurable weights. `None` = flat cosine over the full local
   vector (equal contribution from all parts)

---

## New Parameters

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `local_weight` | `0.0` | `NearestNeighborDistanceMetric` | Local feature distance contribution. `0.0` = disabled |
| `fusion_margin` | `0.05` | `NearestNeighborDistanceMetric` | Half-width of uncertain zone around matching threshold |
| `part_weights` | `None` | `NearestNeighborDistanceMetric` | Per-part weights for local distance. `None` = flat cosine |

---

## Fusion Logic

```
for each track i:
    global_row = cosine_distance(track_gallery, detections)   # always computed

    if local_weight > 0 and track has local gallery:
        uncertain = (threshold - margin) < global_row < (threshold + margin)

        if any uncertain pairs:
            local_dist = part_weighted_cosine(local_gallery, local_features)
            fused = (1 - local_weight) × global_row + local_weight × local_dist
            global_row[uncertain] = fused[uncertain]   # only uncertain pairs updated

    cost_matrix[i] = global_row
```

Clear matches (`global_dist < threshold - margin`) and clear rejects
(`global_dist > threshold + margin`) are never modified by local features.

---

## Backward Compatibility

All changes are additive. The single-feature path is fully preserved:

- `local_feature=None` in Detection → local path never activated
- `local_features=[]` in Track → nothing flushed to local gallery
- `local_weight=0.0` → fusion block skipped entirely
- Any extractor that does not produce a local feature → `local_features=None` passed to
  metric → cost matrix identical to original DeepSORT

---

## Recommended Configuration

### Appearance-primary matching (standard pedestrian sequences)

```python
metric = NearestNeighborDistanceMetric(
    'cosine', 0.30, 100,
    local_weight  = 0.9,
    fusion_margin = 0.15,
    part_weights  = [0.1, 0.4, 0.4, 0.1]  # torso-heavy
)
tracker = Tracker(metric, max_iou_distance=0.3, max_age=30, n_init=3)
```

### Position-primary matching (same-costume / crowded sequences)

```python
metric = NearestNeighborDistanceMetric(
    'cosine', 0.20, 100,
    local_weight  = 0.9,
    fusion_margin = 0.25,
    part_weights  = [0.5, 0.1, 0.1, 0.3]  # head and legs heavy
)
tracker = Tracker(metric, max_iou_distance=0.7, max_age=30, n_init=3)
```

### Disable local features (global-only, original behaviour)

```python
metric = NearestNeighborDistanceMetric('cosine', 0.20, 100, local_weight=0.0)
tracker = Tracker(metric, max_iou_distance=0.7, max_age=30, n_init=3)
```

# Reference
See the original repository [deep_sort](https://github.com/nwojke/deep_sort).
See the [arXiv preprint](https://arxiv.org/abs/1703.07402) for more information about DeepSORT.
