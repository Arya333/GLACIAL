#!/usr/bin/env python
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import networkx

import modeling
import preproc
import plotting
import gtest


def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=999)
    parser.add_argument('--lr', '-l', type=float, default=3e-4)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--reps', type=int, default=4)
    parser.add_argument('--config', '-f', type=pathlib.Path, required=True)
    parser.add_argument('--outpath', '-o', type=pathlib.Path, required=True)
    return parser.parse_args()


def mask_1hot(nb_feat, idx, invert=True):
    if invert:
        return torch.arange(nb_feat) != idx
    return torch.arange(nb_feat) == idx


def full_and_ablated_data(in_):
    dev = in_.device
    out = in_[1:]
    all_in, all_out = [in_], [out]
    T, N, dim = in_.shape  # assume in_.shape[2] == nb_feat
    zero, nan = torch.zeros(1, device=dev), torch.full((1,), torch.nan, device=dev)
    for i in range(dim):
        mask = mask_1hot(dim, i)[None, None, :].to(dev)
        all_in.append(torch.where(mask.tile(T, N, 1), in_, zero))
        all_out.append(torch.where(mask.tile(T-1, N, 1), out, nan))
    return torch.cat(all_in, dim=1), torch.cat(all_out, dim=1),


def self_pred_data(in_):
    dev = in_.device
    out = in_[1:]
    all_in, all_out = [], []
    T, N, dim = in_.shape  # assume in_.shape[2] == nb_feat
    zero, nan = torch.zeros(1, device=dev), torch.full((1,), torch.nan, device=dev)
    for i in range(dim):
        mask = mask_1hot(dim, i, False)[None, None, :].to(dev)
        all_in.append(torch.where(mask.tile(T, N, 1), in_, zero))
        all_out.append(torch.where(mask.tile(T-1, N, 1), out, nan))
    return torch.cat(all_in, dim=1), torch.cat(all_out, dim=1),


def get_mse(model, data):
    D = model.nb_feat
    with torch.no_grad():
        all_in, all_out = full_and_ablated_data(data['in'])
        pred = model.main(all_in, None)
        temporal_avg_mse = F.mse_loss(pred, all_out, reduction='none').nanmean(dim=0).cpu().numpy()
        # temporal_avg_mse *= np.nanstd(all_out.cpu().numpy(), axis=0)  # de-emphasize observation with little variation

    mse = np.array_split(temporal_avg_mse, D+1)
    main_mse = np.nanmean(mse[0], axis=0, keepdims=True)  # average across subjects
    aux_mse = np.full((D, D), np.nan)
    for i, abl_mse_mat in enumerate(mse[1:]):
        t = mask_1hot(D, i)
        aux_mse[i, t] = np.nanmean(abl_mse_mat, axis=0, keepdims=False)[t]  # average across subjects

    with torch.no_grad():
        s_in, s_out = self_pred_data(data['in'])
        pred = model.main(s_in, None)
        s_tempo_avg = F.mse_loss(pred, s_out, reduction='none').nanmean(dim=0).cpu().numpy()

    s_mse = np.full((1, D), np.nan)
    for i, s_mse_mat in enumerate(np.array_split(s_tempo_avg, D)):
        s_mse[0, i] = np.nanmean(s_mse_mat, axis=0)[i]  # average across subjects

    return main_mse, aux_mse, s_mse


def repeat_cv_split(subjects, n_repeat, n_fold):
    for _ in range(n_repeat):
        perm_subjs = np.random.permutation(subjects)
        partitions = np.array_split(perm_subjs, n_fold)
        for i, sub_te in enumerate(partitions):
            j = np.random.choice([idx for idx in range(n_fold) if idx != i])
            sub_va = partitions[j]
            sub_tr = np.concatenate([p for idx, p in enumerate(partitions) if (idx != i) and (idx != j)])
            yield sub_tr, sub_va, sub_te


def gcausality(args, features, exo_feats, frame, subjects):
    args.outpath.mkdir(parents=True, exist_ok=True)
    trace_pairs = []
    mse_pairs = []
    for sub_tr, sub_va, sub_te in repeat_cv_split(subjects, n_repeat=args.reps, n_fold=5):
        frame2 = frame.copy()
        seg_tr = frame2.loc[frame2.RID.apply(lambda x: x in sub_tr), features]
        frame2[features] = (frame2[features] - seg_tr.mean(axis=0)) / seg_tr.std(axis=0)
        d_tr = preproc.create_rnn(frame2, sub_tr, features, exo_feats)
        d_va = preproc.create_rnn(frame2, sub_va, features, exo_feats)
        d_te = preproc.create_rnn(frame2, sub_te, features, exo_feats)

        M = modeling.get_model(len(features), len(exo_feats), len(features)).cuda()
        opt = torch.optim.Adam(M.parameters(), lr=args.lr)
        trace_pairs.append(M.fit(d_tr, d_va, opt, args.steps))
        mse_pairs.append(get_mse(M, d_te))

    plotting.agg_trace(*zip(*trace_pairs)).savefig(args.outpath/f'loss.png', bbox_inches='tight')

    main_mse, aux_mse, s_mse = [np.asarray(l) for l in zip(*mse_pairs)]
    np.savez_compressed(args.outpath/f'mse.npz', main=main_mse, aux=aux_mse, self=s_mse)

    with open(args.config) as fh:
        true_graph = networkx.DiGraph(json.load(fh).get('groundtruth', {}))
    gtest.causal_graph(aux_mse, main_mse, features, args.outpath, true_graph)
    # gtest.causal_graph_v2(aux_mse, main_mse, s_mse, features, args.outpath)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as fh:
        conf = json.load(fh)
    FEATURES = conf['feats']
    EXO_FEATS = []
    frame = pd.read_csv(conf['csv'], usecols=FEATURES + ['RID', 'Year'])

    unique_subjects = frame['RID'].unique()
    unique_subjects = [uid for uid in unique_subjects if (frame['RID'] == uid).sum() > 1]
    print(f'{len(unique_subjects)=}')
    gcausality(args, FEATURES, EXO_FEATS, frame, unique_subjects)


if __name__ == '__main__':
    main(cmd_args())
