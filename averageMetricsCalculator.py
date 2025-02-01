import numpy as np


def calculate_average_metrics(all_accuracy_scores, all_precision_scores, all_recall_scores, all_f1_scores,
                              all_auc_scores):
    average_accuracy = np.mean(all_accuracy_scores)
    average_precision = np.mean(all_precision_scores)
    average_recall = np.mean(all_recall_scores)
    average_f1 = np.mean(all_f1_scores)
    average_auc = np.mean(all_auc_scores)

    return average_accuracy, average_precision, average_recall, average_f1, average_auc
