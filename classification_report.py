def calculate_metrics(actual_labels, predicted_labels, num_classes=10):
    # Initialize counts
    true_positive = [0] * num_classes
    false_positive = [0] * num_classes
    false_negative = [0] * num_classes

    # Populate TP, FP, FN for each class
    for i in range(len(actual_labels)):
        true_label = int(actual_labels[i])
        pred_label = int(predicted_labels[i])
        if true_label == pred_label:
            true_positive[true_label] += 1
        else:
            false_positive[pred_label] += 1
            false_negative[true_label] += 1

    # Calculate precision, recall, and F1-score for each class
    precision = []
    recall = []
    f1_score = []

    for i in range(num_classes):
        tp = true_positive[i]
        fp = false_positive[i]
        fn = false_negative[i]

        # Precision: TP / (TP + FP)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precision.append(p)

        # Recall: TP / (TP + FN)
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall.append(r)

        # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        f1_score.append(f1)

    # Calculate macro-average
    macro_precision = sum(precision) / num_classes
    macro_recall = sum(recall) / num_classes
    macro_f1 = sum(f1_score) / num_classes

    # Print results
    print("Class\tPrecision\tRecall\t\tF1-Score")
    for i in range(num_classes):
        print(f"{i}\t{precision[i]:.2f}\t\t{recall[i]:.2f}\t\t{f1_score[i]:.2f}")

    print("\nMacro-Average Metrics:")
    print(f"Precision: {macro_precision:.2f}")
    print(f"Recall: {macro_recall:.2f}")
    print(f"F1-Score: {macro_f1:.2f}")

    return precision, recall, f1_score, macro_precision, macro_recall, macro_f1




# Assuming act_labels and model_predicted are tensors from the test phase
precision, recall, f1_score, macro_precision, macro_recall, macro_f1 = calculate_metrics(
    act_labels.numpy(), model_predicted.numpy(), num_classes=10
)
