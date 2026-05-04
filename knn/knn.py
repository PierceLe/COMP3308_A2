def classify_knn(training_filename, testing_filename, k):
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

    predictions = []
    for test_row in testing_data:
        distances = []
        for idx, (train_features, train_label) in enumerate(training_data):
            mismatches = 0
            for tf, te in zip(train_features, test_row):
                if tf != te:
                    mismatches += 1
            distance = mismatches ** 0.5
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
