"""Compute information-gain feature ranking and per-classifier confusion
matrices for the report. Outputs a small text file the LaTeX macros block
can quote.
"""
import sys
sys.path.insert(0, '../plus')
sys.path.insert(0, '../knn')
sys.path.insert(0, '../dt')

from plus import _load_labeled, _info_gains, classify_knnplus, classify_dtplus
from knn import classify_knn
from dt import classify_dt


HEART = '../data_processing/heart.csv'
FOLDS = '../stratified-folds/heart-folds.csv'

ATTRS = [
    'age', 'anaemia', 'CPK', 'diabetes', 'ejection_fraction',
    'high_blood_pressure', 'platelets', 'serum_creatinine',
    'serum_sodium', 'sex', 'smoking',
]


def info_gain_ranking():
    rows = _load_labeled(HEART)
    gains = _info_gains(rows)
    ranked = sorted(zip(ATTRS, gains), key=lambda p: -p[1])
    print("Information gain ranking on full heart.csv:")
    for name, g in ranked:
        print(f"  {name:25s}  {g:.4f}")
    return ranked


def parse_folds(path):
    folds = []
    current = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                if current:
                    folds.append(current)
                    current = []
                continue
            if s.lower().startswith('fold'):
                if current:
                    folds.append(current)
                    current = []
                continue
            parts = s.split(',')
            current.append((parts[:-1], parts[-1]))
        if current:
            folds.append(current)
    return folds


def write_fold_files(folds, out_dir='/tmp/cm'):
    import os
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for i, fold in enumerate(folds):
        fp = f"{out_dir}/fold{i}.csv"
        with open(fp, 'w') as f:
            for feats, lbl in fold:
                f.write(','.join(feats) + ',' + lbl + '\n')
        files.append(fp)
    return files


def cv_confusion(classifier_fn, folds, *args, **kwargs):
    """Run 10-fold CV and return aggregated confusion matrix on died class.
    Returns (TP, FN, FP, TN).
    """
    import os, tempfile
    tmpdir = tempfile.mkdtemp()
    fold_files = []
    for i, fold in enumerate(folds):
        fp = f"{tmpdir}/fold{i}.csv"
        with open(fp, 'w') as f:
            for feats, lbl in fold:
                f.write(','.join(feats) + ',' + lbl + '\n')
        fold_files.append(fp)

    TP = FN = FP = TN = 0
    for i in range(len(folds)):
        train_path = f"{tmpdir}/train{i}.csv"
        test_path = f"{tmpdir}/test{i}.csv"
        test_unlab = f"{tmpdir}/testU{i}.csv"
        with open(train_path, 'w') as ftr:
            for j in range(len(folds)):
                if j == i:
                    continue
                with open(fold_files[j]) as fj:
                    for line in fj:
                        ftr.write(line)
        with open(test_path, 'w') as fte, open(test_unlab, 'w') as fteU:
            for feats, lbl in folds[i]:
                fte.write(','.join(feats) + ',' + lbl + '\n')
                fteU.write(','.join(feats) + '\n')
        preds = classifier_fn(train_path, test_unlab, *args, **kwargs)
        truths = [lbl for _, lbl in folds[i]]
        for p, t in zip(preds, truths):
            if t == 'died' and p == 'died':
                TP += 1
            elif t == 'died' and p == 'survived':
                FN += 1
            elif t == 'survived' and p == 'died':
                FP += 1
            else:
                TN += 1
    return TP, FN, FP, TN


def per_fold_acc(classifier_fn, folds, *args, **kwargs):
    """Run 10-fold CV and return list of per-fold accuracies."""
    import tempfile
    tmpdir = tempfile.mkdtemp()
    fold_files = []
    for i, fold in enumerate(folds):
        fp = f"{tmpdir}/fold{i}.csv"
        with open(fp, 'w') as f:
            for feats, lbl in fold:
                f.write(','.join(feats) + ',' + lbl + '\n')
        fold_files.append(fp)

    accs = []
    for i in range(len(folds)):
        train_path = f"{tmpdir}/train{i}.csv"
        test_unlab = f"{tmpdir}/testU{i}.csv"
        with open(train_path, 'w') as ftr:
            for j in range(len(folds)):
                if j == i:
                    continue
                with open(fold_files[j]) as fj:
                    for line in fj:
                        ftr.write(line)
        with open(test_unlab, 'w') as fteU:
            for feats, _ in folds[i]:
                fteU.write(','.join(feats) + '\n')
        preds = classifier_fn(train_path, test_unlab, *args, **kwargs)
        truths = [lbl for _, lbl in folds[i]]
        correct = sum(1 for p, t in zip(preds, truths) if p == t)
        accs.append(correct / len(truths))
    return accs


def metrics_from_cm(TP, FN, FP, TN):
    n = TP + FN + FP + TN
    acc = (TP + TN) / n
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return acc, prec, rec, f1


def paired_t_test(a, b):
    """Paired t-test on lists a and b. Returns (mean_diff, t_stat, df)."""
    import math
    n = len(a)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float('inf')
    return mean, t, n - 1


def main():
    print("=" * 60)
    print("INFORMATION-GAIN FEATURE RANKING")
    print("=" * 60)
    ranking = info_gain_ranking()

    print()
    print("=" * 60)
    print("PER-FOLD ACCURACIES + PAIRED T-TESTS")
    print("=" * 60)
    folds = parse_folds(FOLDS)
    print(f"Loaded {len(folds)} folds")

    knn1 = per_fold_acc(classify_knn, folds, 1)
    knn5 = per_fold_acc(classify_knn, folds, 5)
    knnp1 = per_fold_acc(classify_knnplus, folds, 1)
    knnp5 = per_fold_acc(classify_knnplus, folds, 5)
    dt = per_fold_acc(classify_dt, folds)
    dtp = per_fold_acc(classify_dtplus, folds)

    print(f"MyKNN  k=1   per-fold: {[f'{x:.3f}' for x in knn1]}")
    print(f"MyKNN+ k=1   per-fold: {[f'{x:.3f}' for x in knnp1]}")
    print(f"MyKNN  k=5   per-fold: {[f'{x:.3f}' for x in knn5]}")
    print(f"MyKNN+ k=5   per-fold: {[f'{x:.3f}' for x in knnp5]}")
    print(f"MyDT         per-fold: {[f'{x:.3f}' for x in dt]}")
    print(f"MyDT+        per-fold: {[f'{x:.3f}' for x in dtp]}")

    print()
    for label, a, b in [
        ('MyKNN+ vs MyKNN  k=1', knnp1, knn1),
        ('MyKNN+ vs MyKNN  k=5', knnp5, knn5),
        ('MyDT+  vs MyDT       ', dtp, dt),
    ]:
        mean, t, df = paired_t_test(a, b)
        print(f"{label}: mean diff = {mean:+.4f}  t = {t:+.3f}  df = {df}")

    print()
    print("=" * 60)
    print("CONFUSION MATRICES")
    print("=" * 60)
    for label, fn, args in [
        ('MyKNN  k=1', classify_knn, (1,)),
        ('MyKNN+ k=1', classify_knnplus, (1,)),
        ('MyKNN  k=5', classify_knn, (5,)),
        ('MyKNN+ k=5', classify_knnplus, (5,)),
        ('MyDT       ', classify_dt, ()),
        ('MyDT+      ', classify_dtplus, ()),
    ]:
        TP, FN_, FP, TN = cv_confusion(fn, folds, *args)
        acc, prec, rec, f1 = metrics_from_cm(TP, FN_, FP, TN)
        print(f"{label}: TP={TP:3d}  FN={FN_:3d}  FP={FP:3d}  TN={TN:3d}  "
              f"acc={acc*100:.1f}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")


if __name__ == '__main__':
    main()
