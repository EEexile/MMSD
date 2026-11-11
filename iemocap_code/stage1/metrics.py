"""Emotion classification metrics and WA checkpoint selection."""
import numpy as np


def classification_metrics(labels, predictions, num_classes=4):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    assert len(labels) == len(predictions) and len(labels) > 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    tp = confusion.diagonal()
    recall = np.divide(tp, support, out=np.zeros(num_classes), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(num_classes), where=(support + predicted) > 0)
    correct = int(tp.sum())
    return {'WA': 100.0 * correct / len(labels), 'UA': 100.0 * float(recall.mean()),
            'W-F1': 100.0 * float((f1 * support).sum() / len(labels)),
            'correct': correct, 'n': len(labels), 'confusion': confusion.tolist(),
            'per_class_recall': (100 * recall).tolist(), 'per_class_f1': (100 * f1).tolist()}


def is_better(metrics, best_correct):

    return metrics['correct'] > best_correct
