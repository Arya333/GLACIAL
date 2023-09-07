import numpy as np
import torch
import torch.nn.functional as F


def normalize(frame, training_subjects, features):
    mask = frame['RID'].apply(lambda x: x in training_subjects)
    out = frame.copy()
    out[features] = (out[features] - out.loc[mask, features].mean()) / out.loc[mask, features].std()
    return out


def avg_within_subject(pred, true, rids):
    mse_array = []
    for rid in np.unique(rids):
        smask = rids == rid
        mse_array.append(F.mse_loss(pred[smask], true[smask], reduction='none').nanmean(dim=0, keepdims=True))
    return torch.cat(mse_array, dim=0)


FCI_EDGE = {
    (1, 1): f'o-o',
    (1, 2): f'<-o',
    (1, 3): f'--o',
    (2, 1): f'o->',
    (2, 2): f'<->',
    (2, 3): f'-->',
    (3, 1): f'o--',
    (3, 2): f'<--',
    # (3, 3): f'---',
    (0, 0): None
}
