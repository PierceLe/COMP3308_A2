"""
Build text-based diagrams of MyDT and MyDT+ trained on the full heart.csv.

The diagrams are required as an Appendix in the report.

Run from this folder:
    python3 build_diagrams.py > diagrams.txt
"""

import math
from dt import _build_dt as build_baseline_dt, _print_tree
from plus import (
    _build_dt as build_plus_dt,
    _stratified_split,
    _rep_prune,
    _majority_class,
)


ATTR_NAMES = [
    'age', 'anaemia', 'CPK', 'diabetes', 'ejection_fraction',
    'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium',
    'sex', 'smoking',
]


def load_labeled(filename):
    rows = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            rows.append((parts[:-1], parts[-1]))
    return rows


def count_nodes(tree):
    if tree['leaf']:
        return 1, 1
    leaves = 0
    internals = 1
    for sub in tree['children'].values():
        i, l = count_nodes(sub)
        internals += i - 1 if not sub['leaf'] else 0
        leaves += l
    return internals, leaves


def total_nodes(tree):
    if tree['leaf']:
        return 1
    return 1 + sum(total_nodes(sub) for sub in tree['children'].values())


def num_leaves(tree):
    if tree['leaf']:
        return 1
    return sum(num_leaves(sub) for sub in tree['children'].values())


def main():
    data = load_labeled('heart.csv')
    n_attrs = len(data[0][0])

    print('============================================================')
    print('MyDT - Information Gain decision tree, NO pruning')
    print('Trained on all 273 examples of heart.csv')
    print('============================================================')
    parent_majority = _majority_class([ex[1] for ex in data])
    tree_baseline = build_baseline_dt(data, list(range(n_attrs)), parent_majority)
    _print_tree(tree_baseline, ATTR_NAMES)
    print()
    print(f'Total nodes : {total_nodes(tree_baseline)}')
    print(f'Leaves      : {num_leaves(tree_baseline)}')
    print()

    print('============================================================')
    print('MyDT+ - Information Gain + Reduced-Error Pruning')
    print('Grow set: 75% stratified slice of heart.csv (~205 examples)')
    print('Prune set: remaining 25% (~68 examples)')
    print('============================================================')
    grow, prune = _stratified_split(data, prune_fraction=0.25)
    parent_majority = _majority_class([ex[1] for ex in grow])
    full_tree = build_plus_dt(grow, list(range(n_attrs)), parent_majority)
    pruned = _rep_prune(full_tree, prune)
    _print_tree(pruned, ATTR_NAMES)
    print()
    print(f'Total nodes : {total_nodes(pruned)}')
    print(f'Leaves      : {num_leaves(pruned)}')


if __name__ == '__main__':
    main()
