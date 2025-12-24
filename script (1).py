import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import heapq

# --------------------------
# Utilities
# --------------------------
def sigmoid(x):
    # numerically stable sigmoid
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    return out

def logit(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def kfold_split(X, y, n_splits=5, random_state=None):
    """
    Generate k-fold cross-validation splits from scratch.
    Returns list of (train_idx, test_idx) tuples.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n = X.shape[0]
    idx = np.random.permutation(n)
    
    # Calculate fold sizes
    fold_size = n // n_splits
    remainder = n % n_splits
    
    splits = []
    start_idx = 0
    
    for fold in range(n_splits):
        # Add one extra sample to first 'remainder' folds
        current_fold_size = fold_size + (1 if fold < remainder else 0)
        test_idx = idx[start_idx:start_idx + current_fold_size]
        
        # Training indices are all other indices
        train_mask = np.ones(n, dtype=bool)
        train_mask[start_idx:start_idx + current_fold_size] = False
        train_idx = idx[train_mask]
        
        splits.append((train_idx, test_idx))
        start_idx += current_fold_size
    
    return splits

# --------------------------
# Binning (histogram mapper)
# --------------------------
class BinMapper:
    """
    Quantile-based binning mapper.
    Produces uint8 binned features with at most num_bins distinct bins.
    """
    def __init__(self, num_bins: int = 64):
        if num_bins < 4 or num_bins > 255:
            raise ValueError("num_bins must be in [4, 255]")
        self.num_bins = num_bins
        self.bin_edges_: List[np.ndarray] = []

    def fit(self, X: np.ndarray):
        n_features = X.shape[1]
        self.bin_edges_ = []
        # Use quantiles per feature; ignore NaNs (treat as separate bin if needed)
        for j in range(n_features):
            col = X[:, j]
            # unique safeguards: if too few unique, edges still ok
            qs = np.linspace(0, 1, self.num_bins + 1)[1:-1]
            edges = np.quantile(col, qs, method="linear")
            edges = np.unique(edges)
            self.bin_edges_.append(edges.astype(float))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        Xb = np.empty((n, d), dtype=np.uint8)
        for j in range(d):
            edges = self.bin_edges_[j]
            # np.searchsorted returns bin index in [0, len(edges)]
            Xb[:, j] = np.searchsorted(edges, X[:, j], side="right").astype(np.uint8)
        return Xb

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

# --------------------------
# Internal tree structures
# --------------------------
@dataclass
class _Leaf:
    idxs: np.ndarray         # row indices in this leaf
    best_gain: float = -np.inf
    best_feat: Optional[int] = None
    best_bin: Optional[int] = None
    sum_grad: float = 0.0
    sum_hess: float = 0.0
    node_id: int = -1        # for building final tree
    # children ids set after split:
    left_id: Optional[int] = None
    right_id: Optional[int] = None

@dataclass
class _Node:
    feature: Optional[int]   # None for leaf
    threshold_bin: Optional[int]
    left: Optional[int]      # index in nodes array
    right: Optional[int]
    value: Optional[float]   # leaf value (raw score)

# --------------------------
# One Newton tree (leaf-wise)
# --------------------------
class LGBMTree:
    """
    Single tree trained on gradients/hessians with histogram splits.
    Leaf-wise growth using a priority queue (best-first).
    """
    def __init__(
        self,
        max_leaves: int = 31,
        min_data_in_leaf: int = 20,
        lambda_l2: float = 1.0,
        min_split_gain: float = 0.0,
        feature_fraction: float = 1.0,
        num_bins: int = 64,
        random_state: Optional[int] = None,
    ):
        self.max_leaves = max(2, int(max_leaves))
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.lambda_l2 = float(lambda_l2)
        self.min_split_gain = float(min_split_gain)
        self.feature_fraction = float(np.clip(feature_fraction, 0.0, 1.0))
        self.num_bins = num_bins
        self.random_state = random_state
        self.nodes_: List[_Node] = []
        self.n_features_: Optional[int] = None
        self.feature_subsample_: Optional[np.ndarray] = None

    def _calc_leaf_value(self, G, H):
        # Newton step: -G / (H + λ₂)
        return -G / (H + self.lambda_l2)

    def _split_gain(self, G, H, GL, HL, GR, HR):
        # Gain = 0.5 * [GL^2/(HL+λ) + GR^2/(HR+λ) - G^2/(H+λ)]
        parent = (G * G) / (H + self.lambda_l2)
        left   = (GL * GL) / (HL + self.lambda_l2)
        right  = (GR * GR) / (HR + self.lambda_l2)
        return 0.5 * (left + right - parent)

    def _best_split_for_leaf(self, Xb, grad, hess, leaf: _Leaf, feat_idxs) -> _Leaf:
        idxs = leaf.idxs
        G = np.sum(grad[idxs])
        H = np.sum(hess[idxs])
        leaf.sum_grad, leaf.sum_hess = G, H

        best_gain = -np.inf
        best_feat = None
        best_bin = None

        for f in feat_idxs:
            # build per-bin sums
            xb = Xb[idxs, f]
            # bins are [0..K], where K ≤ num_bins; use bincount up to max bin in this leaf
            maxb = int(xb.max()) if xb.size else 0
            size = max(self.num_bins, maxb + 1)
            G_per_bin = np.bincount(xb, weights=grad[idxs], minlength=size)
            H_per_bin = np.bincount(xb, weights=hess[idxs], minlength=size)

            # prefix sums: consider splits between bins (left includes bins <= t)
            G_prefix = np.cumsum(G_per_bin)
            H_prefix = np.cumsum(H_per_bin)

            # valid splits: ensure both sides have enough data
            # examine all threshold bins t where left_count >= min and right_count >= min
            count_per_bin = np.bincount(xb, minlength=size)
            count_prefix = np.cumsum(count_per_bin)

            # candidate thresholds are 0..size-2 (split between t and t+1)
            # but we use "≤ t" to the left, which is consistent with transform()
            left_counts = count_prefix[:-1]
            right_counts = count_prefix[-1] - count_prefix[:-1]

            valid = (left_counts >= self.min_data_in_leaf) & (right_counts >= self.min_data_in_leaf)
            if not np.any(valid):
                continue

            GL = G_prefix[:-1]
            HL = H_prefix[:-1]
            GR = G - GL
            HR = H - GL

            gains = self._split_gain(G, H, GL, HL, GR, HR)
            gains[~valid] = -np.inf

            t = int(np.argmax(gains))
            gain = float(gains[t])

            if gain > best_gain:
                best_gain = gain
                best_feat = f
                best_bin = t

        leaf.best_gain = best_gain
        leaf.best_feat = best_feat
        leaf.best_bin = best_bin
        return leaf

    def fit(self, X_binned: np.ndarray, grad: np.ndarray, hess: np.ndarray, feature_subsample_mask: Optional[np.ndarray] = None):
        rng = np.random.default_rng(self.random_state)
        n, d = X_binned.shape
        self.n_features_ = d
        self.nodes_ = []

        if feature_subsample_mask is None:
            # choose subset of features once per tree (LightGBM uses per tree)
            if self.feature_fraction < 1.0:
                m = max(1, int(np.ceil(self.feature_fraction * d)))
                feat_idxs = rng.choice(d, size=m, replace=False)
                mask = np.zeros(d, dtype=bool)
                mask[feat_idxs] = True
                self.feature_subsample_ = mask
            else:
                self.feature_subsample_ = np.ones(d, dtype=bool)
        else:
            self.feature_subsample_ = feature_subsample_mask.astype(bool)

        feat_idxs = np.flatnonzero(self.feature_subsample_)

        # Create the root leaf
        root = _Leaf(idxs=np.arange(n), node_id=0)
        root = self._best_split_for_leaf(X_binned, grad, hess, root, feat_idxs)
        # Store nodes in list as we build (index == node_id)
        self.nodes_.append(_Node(feature=None, threshold_bin=None, left=None, right=None, value=None))

        # Priority queue of leaves keyed by -gain (max-heap)
        pq = [(-max(root.best_gain, -np.inf), 0, root)]
        next_node_id = 1
        n_leaves = 1

        while pq and n_leaves < self.max_leaves:
            neg_gain, _, leaf = heapq.heappop(pq)
            gain = -neg_gain
            # stop if no positive gain or min gain threshold not satisfied
            if (gain <= 0.0) or (gain < self.min_split_gain) or (leaf.best_feat is None):
                break

            f = leaf.best_feat
            tbin = leaf.best_bin
            idxs = leaf.idxs
            left_mask = X_binned[idxs, f] <= tbin
            left_idxs = idxs[left_mask]
            right_idxs = idxs[~left_mask]

            # If child too small, skip split
            if (left_idxs.size < self.min_data_in_leaf) or (right_idxs.size < self.min_data_in_leaf):
                break

            # Turn current node into internal split
            node_id = leaf.node_id
            self.nodes_[node_id] = _Node(feature=int(f), threshold_bin=int(tbin), left=next_node_id, right=next_node_id+1, value=None)

            # Create children leaves
            left_leaf = _Leaf(idxs=left_idxs, node_id=next_node_id)
            right_leaf = _Leaf(idxs=right_idxs, node_id=next_node_id+1)

            left_leaf = self._best_split_for_leaf(X_binned, grad, hess, left_leaf, feat_idxs)
            right_leaf = self._best_split_for_leaf(X_binned, grad, hess, right_leaf, feat_idxs)

            # Append placeholder nodes (will set value later if they end up leaves)
            self.nodes_.append(_Node(feature=None, threshold_bin=None, left=None, right=None, value=None))
            self.nodes_.append(_Node(feature=None, threshold_bin=None, left=None, right=None, value=None))

            # Push children into priority queue
            heapq.heappush(pq, (-max(left_leaf.best_gain, -np.inf), left_leaf.node_id, left_leaf))
            heapq.heappush(pq, (-max(right_leaf.best_gain, -np.inf), right_leaf.node_id, right_leaf))

            n_leaves += 1  # we added one extra leaf vs parent

            next_node_id += 2

        # Any remaining nodes in pq (and any not split) become leaves with Newton value
        # First, mark all node ids that already have children
        has_children = np.array([(n.left is not None) and (n.right is not None) for n in self.nodes_], dtype=bool)

        # For all leaves (no children), set value = -G/(H+λ)
        # We need indices set per leaf; collect from pq and also from any not visited (root could have been un-splittable)
        unresolved = {}
        for _, _, leaf in pq:
            unresolved[leaf.node_id] = leaf
        # Also ensure root (or any created leaf) has entry:
        # We reconstruct by scanning nodes_ and building masks on the fly if missing.
        # To avoid O(N * leaves) work, we'll compute G/H for each unresolved leaf using current grad/hess and indices.

        # Build a set of leaf node ids
        leaf_ids = [i for i, node in enumerate(self.nodes_) if not has_children[i]]
        # If a leaf_id missing in unresolved, rebuild its indices by tracing from root (costly);
        # instead we stored indices in the _Leaf objects in pq. For the special case root never split:
        if (0 in leaf_ids) and (0 not in unresolved):
            # root became a leaf
            G = np.sum(grad)
            H = np.sum(hess)
            val = self._calc_leaf_value(G, H)
            self.nodes_[0].value = float(val)

        # For pq leaves we have sums in leaf.sum_grad/hess
        for lid, leaf in unresolved.items():
            val = self._calc_leaf_value(leaf.sum_grad, leaf.sum_hess)
            self.nodes_[lid].value = float(val)

        # Some children may not be in pq if they were never pushed (e.g., we broke early).
        # Any node with no children and still None value: compute from all rows that would land there.
        # To do this efficiently, run a single pass routing all rows and accumulate G/H per leaf node id.
        missing_leaf_ids = [i for i in leaf_ids if self.nodes_[i].value is None]
        if missing_leaf_ids:
            # route every sample to a leaf id
            leaf_for_row = self.apply_leaf_indices(X_binned)
            for lid in missing_leaf_ids:
                rows = (leaf_for_row == lid)
                if not np.any(rows):
                    self.nodes_[lid].value = 0.0
                else:
                    G = float(np.sum(grad[rows]))
                    H = float(np.sum(hess[rows]))
                    self.nodes_[lid].value = float(self._calc_leaf_value(G, H))

        return self

    def _traverse(self, xb_row: np.ndarray) -> float:
        node_id = 0
        while True:
            node = self.nodes_[node_id]
            if node.feature is None:
                return node.value
            if xb_row[node.feature] <= node.threshold_bin:
                node_id = node.left
            else:
                node_id = node.right

    def predict_raw(self, X_binned: np.ndarray) -> np.ndarray:
        return np.array([self._traverse(row) for row in X_binned], dtype=float)

    def apply_leaf_indices(self, X_binned: np.ndarray) -> np.ndarray:
        # Return leaf node_id for each row (used internally)
        out = np.empty(X_binned.shape[0], dtype=int)
        for i, row in enumerate(X_binned):
            node_id = 0
            while True:
                node = self.nodes_[node_id]
                if node.feature is None:
                    out[i] = node_id
                    break
                node_id = node.left if row[node.feature] <= node.threshold_bin else node.right
        return out

# --------------------------
# The Booster (binary classifier)
# --------------------------
class LGBMClassifier:
    """
    LightGBM-style Gradient Boosting for binary classification (logistic loss).
    - Leaf-wise trees with histogram splits
    - Second-order (Newton) updates
    """
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_leaves: int = 31,
        min_data_in_leaf: int = 20,
        lambda_l2: float = 1.0,
        min_split_gain: float = 0.0,
        feature_fraction: float = 1.0,
        bagging_fraction: float = 1.0,
        bagging_freq: int = 0,  # build with bagging every k trees (0 disables)
        num_bins: int = 64,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_leaves = int(max_leaves)
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.lambda_l2 = float(lambda_l2)
        self.min_split_gain = float(min_split_gain)
        self.feature_fraction = float(np.clip(feature_fraction, 0.0, 1.0))
        self.bagging_fraction = float(np.clip(bagging_fraction, 0.0, 1.0))
        self.bagging_freq = int(bagging_freq)
        self.num_bins = int(num_bins)
        self.random_state = random_state

        self._bin_mapper: Optional[BinMapper] = None
        self._trees: List[LGBMTree] = []
        self._base_score: float = 0.0  # raw-score bias (log-odds of prior)
        self.n_features_: Optional[int] = None
        self._rng = np.random.default_rng(random_state)
        
        # Calibration and SHAP attributes
        self.calibration_threshold: float = 0.5
        self.feature_names: Optional[List[str]] = None

    def _compute_grad_hess(self, y: np.ndarray, raw_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = sigmoid(raw_score)
        # Logistic loss grad/hess for y in {0,1}:
        grad = p - y
        hess = np.clip(p * (1.0 - p), 1e-16, None)  # avoid zero
        return grad, hess

    def fit(self, X: np.ndarray, y: np.ndarray):
        y = y.astype(int)
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("LGBMClassifier currently supports binary targets with labels {0,1}.")

        n, d = X.shape
        self.n_features_ = d
        self._bin_mapper = BinMapper(num_bins=self.num_bins).fit(X)
        Xb = self._bin_mapper.transform(X)

        # Prior/base score = log-odds of positive rate (smoothed)
        pos_rate = np.mean(y)
        self._base_score = logit(pos_rate if 0 < pos_rate < 1 else 0.5)

        raw = np.full(n, self._base_score, dtype=float)

        self._trees = []
        for m in range(self.n_estimators):
            # Row bagging
            if (self.bagging_freq > 0) and (m % self.bagging_freq == 0) and (self.bagging_fraction < 1.0):
                idxs = self._rng.choice(n, size=int(np.ceil(self.bagging_fraction * n)), replace=False)
            else:
                idxs = np.arange(n)

            # Grad/Hess on current scores
            grad, hess = self._compute_grad_hess(y[idxs], raw[idxs])

            # Feature subsampling mask (per tree)
            if self.feature_fraction < 1.0:
                mfeat = max(1, int(np.ceil(self.feature_fraction * d)))
                feat_idxs = self._rng.choice(d, size=mfeat, replace=False)
                feat_mask = np.zeros(d, dtype=bool)
                feat_mask[feat_idxs] = True
            else:
                feat_mask = np.ones(d, dtype=bool)

            # Fit one leaf-wise tree on the subset
            tree = LGBMTree(
                max_leaves=self.max_leaves,
                min_data_in_leaf=self.min_data_in_leaf,
                lambda_l2=self.lambda_l2,
                min_split_gain=self.min_split_gain,
                feature_fraction=1.0,      # already handled by feat_mask
                num_bins=self.num_bins,
                random_state=int(self._rng.integers(0, 2**31 - 1)),
            ).fit(Xb[idxs], grad, hess, feature_subsample_mask=feat_mask)

            # Update raw scores for ALL rows (use tree prediction on all X)
            raw += self.learning_rate * tree.predict_raw(Xb)
            self._trees.append(tree)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Xb = self._bin_mapper.transform(X)
        raw = np.full(X.shape[0], self._base_score, dtype=float)
        for tree in self._trees:
            raw += self.learning_rate * tree.predict_raw(Xb)
        return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.decision_function(X)
        p1 = sigmoid(raw)
        return np.vstack([1.0 - p1, p1]).T

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
    
    def set_calibration_threshold(self, X_val: np.ndarray, y_val: np.ndarray, target_specificity: float = 0.90):
        """
        Set threshold to achieve target specificity on validation set.
        """
        pred_proba = self.predict_proba(X_val)[:, 1]
        
        # Get negative samples (y=0) for specificity calculation
        neg_mask = y_val == 0
        if np.sum(neg_mask) == 0:
            raise ValueError("No negative samples in validation set for calibration")
        
        neg_probas = pred_proba[neg_mask]
        
        # FIX: Sort and pick the value at the target_specificity percentile
        sorted_neg_probas = np.sort(neg_probas)
        n_neg = len(sorted_neg_probas)
        
        # To get 90% specificity, we need the 90th percentile of negative scores
        threshold_idx = int(n_neg * target_specificity) 
        threshold_idx = max(0, min(threshold_idx, n_neg - 1))
        
        self.calibration_threshold = sorted_neg_probas[threshold_idx]
        return self.calibration_threshold

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate SHAP-style values for each feature using TreeSHAP approach.
        """
        Xb = self._bin_mapper.transform(X)
        n_samples, n_features = X.shape
        
        # Initialize SHAP values
        shap_values = np.zeros((n_samples, n_features))
        
        # Calculate base value (log-odds of prior)
        base_value = self._base_score
        
        # For each tree, calculate contributions
        for tree in self._trees:
            # Get leaf indices for all samples
            leaf_indices = tree.apply_leaf_indices(Xb)
            
            # For each sample, calculate feature contributions
            for i in range(n_samples):
                # Find path from root to leaf for this sample
                path = self._get_path_to_leaf(tree, Xb[i], leaf_indices[i])
                
                # Calculate contributions for each feature in the path
                for feature_idx, threshold_bin in path:
                    # Approximate contribution as the leaf value of the tree
                    # This is a simplified version of TreeSHAP
                    leaf_node = tree.nodes_[leaf_indices[i]]
                    if leaf_node.value is not None:
                        # Simple approximation: distribute leaf value equally among features in path
                        contribution = leaf_node.value / len(path) if len(path) > 0 else 0
                        shap_values[i, feature_idx] += self.learning_rate * contribution
        
        return shap_values
    
    def _get_path_to_leaf(self, tree, x_row, target_leaf_id):
        """
        Get the path from root to a specific leaf node.
        """
        path = []
        node_id = 0
        while True:
            node = tree.nodes_[node_id]
            if node.feature is None:  # Leaf node
                return path
            if x_row[node.feature] <= node.threshold_bin:
                path.append((node.feature, node.threshold_bin))
                node_id = node.left
            else:
                path.append((node.feature, node.threshold_bin))
                node_id = node.right

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n = X.shape[0]
    idx = np.random.permutation(n)
    n_test = int(n * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

