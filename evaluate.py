from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np


def evaluate_model_in(model, test_features, test_labels):
    predictions = model.predict(test_features)
    binary_predictions = np.round(predictions)
    binary_predictions = np.argmax(binary_predictions, axis=1)
    accuracy = accuracy_score(test_labels, binary_predictions)
    precision = precision_score(test_labels, binary_predictions)
    recall = recall_score(test_labels, binary_predictions)
    f1 = f1_score(test_labels, binary_predictions)
    confusion = confusion_matrix(test_labels, binary_predictions)
    fpr, tpr, _ = roc_curve(test_labels, predictions[:, 1])
    roc_auc = auc(fpr, tpr)
    return accuracy, precision, recall, f1, confusion, roc_auc, fpr, tpr


def evaluate_model(model, test_features, test_labels):
    predictions = model.predict(test_features)
    binary_predictions = np.round(predictions)
    accuracy = accuracy_score(test_labels, binary_predictions)
    precision = precision_score(test_labels, binary_predictions)
    recall = recall_score(test_labels, binary_predictions)
    f1 = f1_score(test_labels, binary_predictions)
    confusion = confusion_matrix(test_labels, binary_predictions)
    fpr, tpr, _ = roc_curve(test_labels, predictions)
    roc_auc = auc(fpr, tpr)
    return accuracy, precision, recall, f1, confusion, roc_auc, fpr, tpr
