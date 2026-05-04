import math


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
            v = ex[0][attr]
            partitions.setdefault(v, []).append(ex)

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


def classify_dt(training_filename, testing_filename):
    training_data = []
    with open(training_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            features = parts[:-1]
            label = parts[-1]
            training_data.append((features, label))

    testing_data = []
    with open(testing_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            testing_data.append(parts)

    if not training_data:
        return ['died'] * len(testing_data)

    n_attributes = len(training_data[0][0])
    attribute_indices = list(range(n_attributes))
    parent_majority = _majority_class([ex[1] for ex in training_data])

    tree = _build_dt(training_data, attribute_indices, parent_majority)

    return [_predict_dt(tree, inst) for inst in testing_data]


def _print_tree(tree, attribute_names=None, indent=0):
    prefix = '  ' * indent
    if tree['leaf']:
        print(f"{prefix}-> {tree['class']}")
        return
    attr = tree['attr']
    name = attribute_names[attr] if attribute_names else f"attr[{attr}]"
    for value, subtree in tree['children'].items():
        print(f"{prefix}{name} = {value}")
        _print_tree(subtree, attribute_names, indent + 1)


def classify_dt_with_tree(training_filename, testing_filename, attribute_names=None):
    training_data = []
    with open(training_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            training_data.append((parts[:-1], parts[-1]))

    testing_data = []
    with open(testing_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            testing_data.append(line.split(','))

    n_attributes = len(training_data[0][0])
    parent_majority = _majority_class([ex[1] for ex in training_data])
    tree = _build_dt(training_data, list(range(n_attributes)), parent_majority)

    print("=== Decision Tree ===")
    _print_tree(tree, attribute_names)
    print("=====================")

    return [_predict_dt(tree, inst) for inst in testing_data]
