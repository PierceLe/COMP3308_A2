"""
Formal 10-fold stratified cross-validation using heart-folds.csv.

Reports per-fold and average accuracy, precision, recall, F1 (per class and
macro-averaged) and confusion matrices for the four classifiers we built:

    MyKNN     - knn.classify_knn  (Hamming + simple majority)
    MyKNN+    - plus.classify_knnplus  (info-gain weighted VDM)
    MyDT      - dt.classify_dt
    MyDT+     - plus.classify_dtplus  (Information Gain + REP)

Run from this folder:
    python3 evaluate.py
"""

from knn import classify_knn
from dt import classify_dt
from plus import classify_knnplus, classify_dtplus


FOLDS_FILE = 'heart-folds.csv'


def load_folds(filename):
    """Parse heart-folds.csv into list of folds (each fold = list of 'features...,class' lines)."""
    folds = []
    current = None
    with open(filename, 'r') as f:
        for raw in f:
            line = raw.rstrip('\n').rstrip('\r')
            if not line.strip():
                if current is not None:
                    folds.append(current)
                    current = None
                continue
            if line.startswith('fold'):
                if current is not None:
                    folds.append(current)
                current = []
            else:
                if current is not None:
                    current.append(line.strip())
    if current is not None:
        folds.append(current)
    return folds


def write_csv(rows, filename, drop_class=False):
    with open(filename, 'w') as f:
        for row in rows:
            if drop_class:
                row = ','.join(row.split(',')[:-1])
            f.write(row + '\n')


def confusion(y_true, y_pred, positive='died'):
    """Returns (TP, FP, FN, TN) treating 'died' as positive class."""
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == positive and p == positive:
            tp += 1
        elif t != positive and p == positive:
            fp += 1
        elif t == positive and p != positive:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def safe_div(a, b):
    return a / b if b > 0 else 0.0


def per_class_metrics(y_true, y_pred, classes=('died', 'survived')):
    """Return dict: class -> {precision, recall, f1, support}."""
    out = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        f1 = safe_div(2 * prec * rec, prec + rec)
        sup = sum(1 for t in y_true if t == c)
        out[c] = {'precision': prec, 'recall': rec, 'f1': f1, 'support': sup}
    return out


def cv_evaluate(classify_fn, folds, **kwargs):
    """Returns aggregated y_true and y_pred across all folds, plus per-fold accuracies."""
    all_true, all_pred = [], []
    fold_acc = []
    n = len(folds)
    for i in range(n):
        train_rows = [r for j, fold in enumerate(folds) if j != i for r in fold]
        test_rows = folds[i]
        write_csv(train_rows, '_train.csv')
        write_csv(test_rows, '_test.csv', drop_class=True)
        true = [r.split(',')[-1] for r in test_rows]
        pred = classify_fn('_train.csv', '_test.csv', **kwargs)
        assert len(pred) == len(true), f'pred/true length mismatch in fold {i+1}'
        all_true.extend(true)
        all_pred.extend(pred)
        fold_acc.append(sum(1 for t, p in zip(true, pred) if t == p) / len(true))
    return all_true, all_pred, fold_acc


def report(name, all_true, all_pred, fold_acc):
    n = len(all_true)
    correct = sum(1 for t, p in zip(all_true, all_pred) if t == p)
    overall = correct / n
    mean_acc = sum(fold_acc) / len(fold_acc)
    var = sum((a - mean_acc) ** 2 for a in fold_acc) / len(fold_acc)
    sd = var ** 0.5
    tp, fp, fn, tn = confusion(all_true, all_pred, positive='died')
    pcm = per_class_metrics(all_true, all_pred)
    macro_p = (pcm['died']['precision'] + pcm['survived']['precision']) / 2
    macro_r = (pcm['died']['recall'] + pcm['survived']['recall']) / 2
    macro_f = (pcm['died']['f1'] + pcm['survived']['f1']) / 2

    print(f'\n=== {name} ===')
    print(f'  Per-fold accuracy: ' + ' '.join(f'{a:.3f}' for a in fold_acc))
    print(f'  Mean fold accuracy = {mean_acc:.4f}  SD = {sd:.4f}')
    print(f'  Overall (pooled) accuracy = {overall:.4f}  ({correct}/{n})')
    print(f'  Confusion matrix (positive=died):')
    print(f'                     pred died    pred survived')
    print(f'    actual died        {tp:5d}         {fn:5d}')
    print(f'    actual survived    {fp:5d}         {tn:5d}')
    print(f'  Per-class metrics:')
    for c in ('died', 'survived'):
        m = pcm[c]
        print(f'    {c:9s}: P={m["precision"]:.4f}  R={m["recall"]:.4f}  F1={m["f1"]:.4f}  support={m["support"]}')
    print(f'  Macro-avg : P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f:.4f}')
    return {
        'name': name,
        'mean_acc': mean_acc,
        'sd_acc': sd,
        'overall_acc': overall,
        'fold_acc': fold_acc,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'pcm': pcm,
        'macro_p': macro_p, 'macro_r': macro_r, 'macro_f': macro_f,
    }


def main():
    folds = load_folds(FOLDS_FILE)
    print(f'Loaded {len(folds)} folds; total examples = {sum(len(f) for f in folds)}')

    results = []
    for k in (1, 5):
        results.append(report(f'MyKNN k={k}',
                              *cv_evaluate(classify_knn, folds, k=k)))
        results.append(report(f'MyKNN+ k={k}',
                              *cv_evaluate(classify_knnplus, folds, k=k)))
    results.append(report('MyDT',  *cv_evaluate(classify_dt,  folds)))
    results.append(report('MyDT+', *cv_evaluate(classify_dtplus, folds)))

    print('\n=== Summary table (mean fold accuracy) ===')
    for r in results:
        print(f'  {r["name"]:<12} acc={r["mean_acc"]:.4f}  '
              f'F1(died)={r["pcm"]["died"]["f1"]:.4f}  '
              f'Recall(died)={r["pcm"]["died"]["recall"]:.4f}')

    import os
    for fp in ('_train.csv', '_test.csv'):
        if os.path.exists(fp):
            os.remove(fp)


if __name__ == '__main__':
    main()
