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

# Deep SORT

## Introduction

This repository contains code for *Simple Online and Realtime Tracking with a Deep Association Metric* (Deep SORT).
We extend the original [SORT](https://github.com/abewley/sort) algorithm to
integrate appearance information based on a deep appearance descriptor.
See the [arXiv preprint](https://arxiv.org/abs/1703.07402) for more information.

## Installation

First, clone the repository and install dependencies:
```
git clone https://github.com/nwojke/deep_sort.git
cd deep_sort

# The following command installs all the dependencies required to run the
# tracker and regenerate detections. If you only need to run the tracker with
# existing detections, you can use pip install -r requirements.txt instead.
pip install -r requirements-gpu.txt
```
Then, download pre-generated detections and the CNN checkpoint file from
[here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp).

*NOTE:* The candidate object locations of our pre-generated detections are
taken from the following paper:
```
F. Yu, W. Li, Q. Li, Y. Liu, X. Shi, J. Yan. POI: Multiple Object Tracking with
High Performance Detection and Appearance Feature. In BMTT, SenseTime Group
Limited, 2016.
```
We have replaced the appearance descriptor with a custom deep convolutional
neural network (see below).

## Running the tracker

The following example starts the tracker on one of the
[MOT16 benchmark](https://motchallenge.net/data/MOT16/)
sequences.
We assume resources have been extracted to the repository root directory and
the MOT16 benchmark data is in `./MOT16`:
```
python deep_sort_app.py \
    --sequence_dir=./MOT16/test/MOT16-06 \
    --detection_file=./resources/detections/MOT16_POI_test/MOT16-06.npy \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True
```
Check `python deep_sort_app.py -h` for an overview of available options.
There are also scripts in the repository to visualize results, generate videos,
and evaluate the MOT challenge benchmark.

## Generating detections

Beside the main tracking application, this repository contains a script to
generate features for person re-identification, suitable to compare the visual
appearance of pedestrian bounding boxes using cosine similarity.
The following example generates these features from standard MOT challenge
detections. Again, we assume resources have been extracted to the repository
root directory and MOT16 data is in `./MOT16`:
```
python tools/generate_detections.py \
    --model=resources/networks/mars-small128.pb \
    --mot_dir=./MOT16/train \
    --output_dir=./resources/detections/MOT16_train
```
The model has been generated with TensorFlow 1.5. If you run into
incompatibility, re-export the frozen inference graph to obtain a new
`mars-small128.pb` that is compatible with your version:
```
python tools/freeze_model.py
```
The ``generate_detections.py`` stores for each sequence of the MOT16 dataset
a separate binary file in NumPy native format. Each file contains an array of
shape `Nx138`, where N is the number of detections in the corresponding MOT
sequence. The first 10 columns of this array contain the raw MOT detection
copied over from the input file. The remaining 128 columns store the appearance
descriptor. The files generated by this command can be used as input for the
`deep_sort_app.py`.

**NOTE**: If ``python tools/generate_detections.py`` raises a TensorFlow error,
try passing an absolute path to the ``--model`` argument. This might help in
some cases.

## Training the model

To train the deep association metric model we used a novel [cosine metric learning](https://github.com/nwojke/cosine_metric_learning) approach which is provided as a separate repository.

## Highlevel overview of source files

In the top-level directory are executable scripts to execute, evaluate, and
visualize the tracker. The main entry point is in `deep_sort_app.py`.
This file runs the tracker on a MOTChallenge sequence.

In package `deep_sort` is the main tracking code:

* `detection.py`: Detection base class.
* `kalman_filter.py`: A Kalman filter implementation and concrete
   parametrization for image space filtering.
* `linear_assignment.py`: This module contains code for min cost matching and
   the matching cascade.
* `iou_matching.py`: This module contains the IOU matching metric.
* `nn_matching.py`: A module for a nearest neighbor matching metric.
* `track.py`: The track class contains single-target track data such as Kalman
  state, number of hits, misses, hit streak, associated feature vectors, etc.
* `tracker.py`: This is the multi-target tracker class.

The `deep_sort_app.py` expects detections in a custom format, stored in .npy
files. These can be computed from MOTChallenge detections using
`generate_detections.py`. We also provide
[pre-generated detections](https://drive.google.com/open?id=1VVqtL0klSUvLnmBKS89il1EKC3IxUBVK).

## Citing DeepSORT

If you find this repo useful in your research, please consider citing the following papers:

    @inproceedings{Wojke2017simple,
      title={Simple Online and Realtime Tracking with a Deep Association Metric},
      author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
      booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
      year={2017},
      pages={3645--3649},
      organization={IEEE},
      doi={10.1109/ICIP.2017.8296962}
    }

    @inproceedings{Wojke2018deep,
      title={Deep Cosine Metric Learning for Person Re-identification},
      author={Wojke, Nicolai and Bewley, Alex},
      booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
      year={2018},
      pages={748--756},
      organization={IEEE},
      doi={10.1109/WACV.2018.00087}
    }
