import numpy as np


def majority_vote_predictions(vote_buffer):
    preds = [item[0] for item in vote_buffer if item[0] is not None]

    if len(preds) < 2:
        return None

    n_conscious = sum(1 for p in preds if int(p) == 1)
    n_unconscious = sum(1 for p in preds if int(p) == 0)

    if n_conscious > n_unconscious:
        return 1
    elif n_unconscious > n_conscious:
        return 0
    else:
        return None


def average_probability(vote_buffer):
    probs = [item[1] for item in vote_buffer if item[1] is not None]
    if len(probs) == 0:
        return None
    return float(np.mean(probs))


def average_latency(vote_buffer):
    lats = [item[3] for item in vote_buffer]
    if len(lats) == 0:
        return 0.0
    return float(np.mean(lats))