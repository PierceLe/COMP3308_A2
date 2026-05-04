"""
Generate heart-folds.csv : 10 stratified folds of heart.csv.

Algorithm (deterministic, no shuffling).
    1. Group every row of heart.csv by its class label ('died' / 'survived').
    2. Distribute each class's rows round-robin across the 10 folds, in the
       order they appear in heart.csv. This guarantees:
        * fold sizes differ by at most one
        * the class ratio per fold matches the global ratio as closely as
          possible (each class is split evenly across folds)
    3. Write the folds to heart-folds.csv in the required Ed format:
            fold1
            <rows of fold 1>
            <blank line>
            fold2
            <rows of fold 2>
            ...
            fold10
            <rows of fold 10>

Run from this folder:
    python3 make_folds.py
"""

N_FOLDS = 10
INPUT = 'heart.csv'
OUTPUT = 'heart-folds.csv'


def main():
    by_class = {}
    with open(INPUT, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls = line.split(',')[-1]
            by_class.setdefault(cls, []).append(line)

    folds = [[] for _ in range(N_FOLDS)]
    for cls in sorted(by_class):
        for i, row in enumerate(by_class[cls]):
            folds[i % N_FOLDS].append(row)

    with open(OUTPUT, 'w') as f:
        for i, fold in enumerate(folds):
            f.write(f'fold{i + 1}\n')
            for row in fold:
                f.write(row + '\n')
            if i < N_FOLDS - 1:
                f.write('\n')

    total = sum(len(fold) for fold in folds)
    print(f'Wrote {OUTPUT} with {total} examples in {N_FOLDS} folds.')
    print(f'{"Fold":<8} {"died":<6} {"survived":<10} {"total":<6} {"%died":<6}')
    for i, fold in enumerate(folds):
        d = sum(1 for r in fold if r.endswith('died'))
        s = sum(1 for r in fold if r.endswith('survived'))
        pct = d / (d + s) * 100
        print(f'fold{i + 1:<4} {d:<6} {s:<10} {d + s:<6} {pct:<6.2f}')


if __name__ == '__main__':
    main()
