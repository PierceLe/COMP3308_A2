"""
KNN+ and DT+ : variations of the baseline classifiers for the heart-failure
survival prediction task.

KNN+ improvements (over baseline KNN with Hamming distance + simple majority).
    1. Value Difference Metric (VDM, Stanfill & Waltz 1986).
       The baseline treats every nominal mismatch as equally distant (1.0),
       ignoring that some values are semantically close (e.g. CPK = very-high
       vs CPK = severely-high) while others are far apart. VDM defines
            delta_a(v1, v2) = sum_c |P(c | a = v1) - P(c | a = v2)|
       Conditional probabilities are estimated from the training set with
       Laplace smoothing so unseen (value, class) pairs do not collapse to
       zero. Test-time values not seen during training fall back to the
       class prior.
    2. Information-gain feature weighting.
       Hamming and plain VDM weight every attribute equally even though the
       heart-failure features have very different predictive power. Each
       attribute's squared VDM delta is multiplied by that attribute's
       information gain so highly informative attributes dominate the
       distance.

    Voting is kept as simple majority (with tie -> died). Empirical 10-fold
    CV showed that distance-weighted voting (1 / (d + eps)) lets a single
    near-zero-distance neighbour dominate, which cancels the smoothing
    benefit of larger k and hurts mean accuracy.

DT+ improvement (over baseline DT with Information Gain and no pruning).
    Reduced-Error Pruning (Quinlan 1987).
        a. Stratify-split the training data into a "grow" set (~80%) and a
           "prune" set (~20%). The split is deterministic so the function
           is reproducible.
        b. Build a full Information-Gain decision tree on the grow set.
        c. Walk the tree bottom-up. For every internal node compare the
           accuracy on the prune set of (i) keeping the subtree vs.
           (ii) replacing it with a leaf labelled with the node's
           training-majority class. If the leaf is at least as accurate,
           prune.
    Motivation: the unpruned tree reaches 100% training accuracy on 273
    examples - a textbook symptom of overfitting which REP directly removes.

The original assignment tie-breaking rules are preserved everywhere:
    * KNN distance ties -> neighbour with the lower training row index wins
    * any class tie     -> predict 'died'
"""

import math


# --------------------------------------------------------------------------- #
# Shared I/O helpers
# --------------------------------------------------------------------------- #

def _load_labeled(filename):
    rows = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            rows.append((parts[:-1], parts[-1]))
    return rows


def _load_unlabeled(filename):
    rows = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(line.split(','))
    return rows


def _entropy(labels):
    if not labels:
        return 0.0
    total = len(labels)
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def _majority_class(labels):
    if not labels:
        return 'died'
    died = 0
    survived = 0
    for l in labels:
        if l == 'died':
            died += 1
        elif l == 'survived':
            survived += 1
    if died >= survived:
        return 'died'
    return 'survived'


# --------------------------------------------------------------------------- #
# KNN+
# --------------------------------------------------------------------------- #

def _info_gains(training_data):
    """Information gain of every attribute given the training class labels."""
    n_attrs = len(training_data[0][0])
    base = _entropy([ex[1] for ex in training_data])
    total = len(training_data)
    gains = []
    for a in range(n_attrs):
        partitions = {}
        for ex in training_data:
            partitions.setdefault(ex[0][a], []).append(ex[1])
        weighted = 0.0
        for sub in partitions.values():
            weighted += (len(sub) / total) * _entropy(sub)
        gains.append(max(0.0, base - weighted))
    return gains


def _build_vdm_tables(training_data):
    """Estimate P(class | attribute = value) with Laplace smoothing."""
    n_attrs = len(training_data[0][0])
    classes = sorted({label for _, label in training_data})
    k_classes = len(classes)

    value_class_counts = [{} for _ in range(n_attrs)]
    value_totals = [{} for _ in range(n_attrs)]

    for features, label in training_data:
        for a, v in enumerate(features):
            if v not in value_class_counts[a]:
                value_class_counts[a][v] = {c: 0 for c in classes}
            value_class_counts[a][v][label] += 1
            value_totals[a][v] = value_totals[a].get(v, 0) + 1

    p_table = [{} for _ in range(n_attrs)]
    for a in range(n_attrs):
        for v, counts in value_class_counts[a].items():
            n_v = value_totals[a][v]
            p_table[a][v] = {c: (counts[c] + 1) / (n_v + k_classes) for c in classes}

    class_counts = {c: 0 for c in classes}
    for _, label in training_data:
        class_counts[label] += 1
    total = sum(class_counts.values())
    class_prior = {c: class_counts[c] / total for c in classes}

    return p_table, class_prior, classes


def classify_knnplus(training_filename, testing_filename, k):
    training_data = _load_labeled(training_filename)
    testing_data = _load_unlabeled(testing_filename)

    if not training_data:
        return ['died'] * len(testing_data)

    p_table, class_prior, classes = _build_vdm_tables(training_data)
    gains = _info_gains(training_data)

    predictions = []

    for test_row in testing_data:
        distances = []
        for idx, (train_features, train_label) in enumerate(training_data):
            d_sq = 0.0
            for a, (tf, te) in enumerate(zip(train_features, test_row)):
                p1 = p_table[a].get(tf, class_prior)
                p2 = p_table[a].get(te, class_prior)
                delta = 0.0
                for c in classes:
                    delta += abs(p1[c] - p2[c])
                d_sq += gains[a] * delta * delta
            distance = d_sq ** 0.5
            distances.append((distance, idx, train_label))

        distances.sort(key=lambda x: (x[0], x[1]))
        top_k = distances[:k]

        died_count = 0
        survived_count = 0
        for _, _, lbl in top_k:
            if lbl == 'died':
                died_count += 1
            elif lbl == 'survived':
                survived_count += 1

        if died_count >= survived_count:
            predictions.append('died')
        else:
            predictions.append('survived')

    return predictions


# --------------------------------------------------------------------------- #
# DT+
# --------------------------------------------------------------------------- #

def _build_dt(examples, attribute_indices, parent_majority):
    if not examples:
        return {'leaf': True, 'class': parent_majority}

    labels = [ex[1] for ex in examples]
    if len(set(labels)) == 1:
        return {'leaf': True, 'class': labels[0]}
    if not attribute_indices:
        return {'leaf': True, 'class': _majority_class(labels)}

    base_entropy = _entropy(labels)
    total = len(examples)

    best_attr = None
    best_gain = -1.0
    best_partitions = None

    for attr in attribute_indices:
        partitions = {}
        for ex in examples:
            partitions.setdefault(ex[0][attr], []).append(ex)

        weighted_entropy = 0.0
        for subset in partitions.values():
            sub_labels = [e[1] for e in subset]
            weighted_entropy += (len(subset) / total) * _entropy(sub_labels)

        gain = base_entropy - weighted_entropy
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
            best_partitions = partitions

    node_majority = _majority_class(labels)

    if best_attr is None or best_gain <= 0:
        return {'leaf': True, 'class': node_majority}

    remaining = [a for a in attribute_indices if a != best_attr]
    children = {}
    for v, subset in best_partitions.items():
        children[v] = _build_dt(subset, remaining, node_majority)

    return {
        'leaf': False,
        'attr': best_attr,
        'majority': node_majority,
        'children': children,
    }


def _predict_dt(tree, instance):
    node = tree
    while not node['leaf']:
        attr = node['attr']
        v = instance[attr]
        if v not in node['children']:
            return node['majority']
        node = node['children'][v]
    return node['class']


def _stratified_split(training_data, prune_fraction=0.2):
    by_class = {}
    for ex in training_data:
        by_class.setdefault(ex[1], []).append(ex)

    grow_set, prune_set = [], []
    for c, exs in by_class.items():
        n_prune = max(1, int(round(len(exs) * prune_fraction)))
        n_prune = min(n_prune, len(exs) - 1) if len(exs) > 1 else 0
        prune_set.extend(exs[-n_prune:] if n_prune > 0 else [])
        grow_set.extend(exs[:-n_prune] if n_prune > 0 else exs)
    return grow_set, prune_set


def _rep_prune(tree, prune_examples):
    """Bottom-up reduced-error pruning."""
    if tree['leaf']:
        return tree

    attr = tree['attr']
    new_children = {}
    for v, subtree in tree['children'].items():
        sub_prune = [ex for ex in prune_examples if ex[0][attr] == v]
        new_children[v] = _rep_prune(subtree, sub_prune)
    tree = {
        'leaf': False,
        'attr': attr,
        'majority': tree['majority'],
        'children': new_children,
    }

    if not prune_examples:
        return tree

    correct_subtree = sum(
        1 for ex in prune_examples if _predict_dt(tree, ex[0]) == ex[1]
    )
    correct_leaf = sum(1 for ex in prune_examples if tree['majority'] == ex[1])

    if correct_leaf >= correct_subtree:
        return {'leaf': True, 'class': tree['majority']}
    return tree


def classify_dtplus(training_filename, testing_filename):
    training_data = _load_labeled(training_filename)
    testing_data = _load_unlabeled(testing_filename)

    if not training_data:
        return ['died'] * len(testing_data)

    grow_set, prune_set = _stratified_split(training_data, prune_fraction=0.25)

    n_attrs = len(training_data[0][0])
    parent_majority = _majority_class([ex[1] for ex in grow_set])
    tree = _build_dt(grow_set, list(range(n_attrs)), parent_majority)

    if prune_set:
        tree = _rep_prune(tree, prune_set)

    return [_predict_dt(tree, inst) for inst in testing_data]
