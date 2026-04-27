# vim: expandtab:ts=4:sw=4
import numpy as np


def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


def _nn_cosine_distance_parts(x, y, part_weights, part_dim=768):
    """Nearest-neighbor cosine distance with per-part weighting for JPM features.

    Parameters
    ----------
    x : ndarray
        An NxK gallery matrix (K = num_parts x part_dim, e.g. 3072).
    y : ndarray
        An MxK query matrix.
    part_weights : array_like
        Weights for each part (must sum to 1.0). Length = num_parts.
        Order: [head, upper_torso, lower_torso, legs].
    part_dim : int
        Dimensionality of each part vector. Default 768 (TransReID ViT-Base).

    Returns
    -------
    ndarray
        A vector of length M — nearest-neighbour weighted part distance per query.
    """
    x, y = np.asarray(x), np.asarray(y)
    num_parts = len(part_weights)
    part_weights = np.asarray(part_weights, dtype=np.float32)
    # reshape to (N, num_parts, part_dim) and (M, num_parts, part_dim)
    x_parts = x.reshape(len(x), num_parts, part_dim)
    y_parts = y.reshape(len(y), num_parts, part_dim)
    # normalize each part independently
    x_parts = x_parts / (np.linalg.norm(x_parts, axis=2, keepdims=True) + 1e-8)
    y_parts = y_parts / (np.linalg.norm(y_parts, axis=2, keepdims=True) + 1e-8)
    # cosine distance per part: shape (N, M, num_parts)
    part_dists = 1.0 - np.einsum('npd,mpd->nmp', x_parts, y_parts)
    # weighted sum over parts: shape (N, M)
    weighted = np.dot(part_dists, part_weights)
    # nearest neighbour over gallery (min over N): shape (M,)
    return weighted.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, matching_threshold, budget=None,
                 local_weight=0.0, fusion_margin=0.05, part_weights=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}
        # local_weight: fraction of distance from JPM local features (0.0 = global only)
        # e.g. 0.3 → final_dist = 0.7 * global_dist + 0.3 * local_dist
        self.local_weight = local_weight
        # fusion_margin: JPM only activates when global_dist is within ±margin of threshold.
        # Prevents JPM from overriding confident global decisions (clear match or clear reject).
        # 0.0 → JPM never activates; 0.05 → tight uncertain zone (recommended); 0.10 → wider
        self.fusion_margin = fusion_margin
        # part_weights: per-part weights for JPM local distance [head, upper, lower, legs].
        # None → flat 3072-dim cosine (equal 25% per part, same as before).
        # [0.4, 0.2, 0.2, 0.2] → head-heavy, best for same-costume datasets.
        self.part_weights  = part_weights
        self.local_samples = {}  # Dict[track_id -> List[ndarray(3072,)]]

    def partial_fit(self, features, targets, active_targets, local_features=None):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.
        local_features : Optional[ndarray]
            An NxK matrix of N local features (e.g. 3072-dim JPM parts).
            If None, local distance fusion is skipped.

        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

        if local_features is not None:
            for lf, target in zip(local_features, targets):
                self.local_samples.setdefault(target, []).append(lf)
                if self.budget is not None:
                    self.local_samples[target] = self.local_samples[target][-self.budget:]
        self.local_samples = {k: self.local_samples[k] for k in active_targets
                              if k in self.local_samples}

    def distance(self, features, targets, local_features=None):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.
        local_features : Optional[ndarray]
            An NxK matrix of N local features for distance-level fusion.
            If None or local_weight==0, global-only distance is returned.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)

        # Distance-level fusion: per-track — fuse only tracks that have local gallery,
        # fall back to global-only for tracks that don't (e.g. newly confirmed tracks).
        # Previous all() guard caused fusion to NEVER activate because there is always
        # at least one newly confirmed track without local_samples in the batch.
        if (local_features is not None
                and self.local_weight > 0.0
                and len(self.local_samples) > 0):
            lo = max(0.0, self.matching_threshold - self.fusion_margin)
            hi = min(1.0, self.matching_threshold + self.fusion_margin)
            for i, target in enumerate(targets):
                if target in self.local_samples:
                    global_row = cost_matrix[i, :]
                    uncertain  = (global_row > lo) & (global_row < hi)
                    if uncertain.any():
                        if self.part_weights is not None:
                            local_dist = _nn_cosine_distance_parts(
                                self.local_samples[target], local_features, self.part_weights)
                        else:
                            local_dist = self._metric(self.local_samples[target], local_features)
                        fused = ((1.0 - self.local_weight) * global_row
                                 + self.local_weight * local_dist)
                        cost_matrix[i, uncertain] = fused[uncertain]
                    # clear decisions (global_dist outside uncertain zone) → unchanged
                # else: no local gallery for this track yet → keep global-only row

        return cost_matrix
