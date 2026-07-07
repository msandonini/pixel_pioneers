import numpy as np
from scipy.stats import pearsonr, spearmanr


def plcc(predictions, targets):
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    score, _ = pearsonr(predictions, targets)
    return score


def srocc(predictions, targets):
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    score, _ = spearmanr(predictions, targets)
    return score


def compute_metrics(predictions, targets):
    return {
        "PLCC": plcc(predictions, targets),
        "SROCC": srocc(predictions, targets),
    }