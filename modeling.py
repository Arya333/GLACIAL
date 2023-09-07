import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats as stats
import util

import model


def mask_1hot(nb_feat, idx, invert=True):
    if invert:
        return torch.arange(nb_feat) != idx
    return torch.arange(nb_feat) == idx


def _test_mse_one2one(nb_feat, main_mse, stack):
    rep, D = len(main_mse), main_mse[0].shape[-1]
    out_st = np.full((rep, D, D), np.nan)
    out_pv = np.full((rep, D, D), np.nan)

    for i, aux_mse in enumerate(stack):
        fmask = mask_1hot(nb_feat, i)
        st_mat, pv_mat = [], []
        for ma, mb in zip(main_mse, aux_mse):
            st, pv = stats.ttest_rel(ma[:, fmask], mb, axis=0, nan_policy='omit')
            st_mat.append(st)
            pv_mat.append(pv)
        out_st[:, i, fmask] = np.vstack(st_mat)
        out_pv[:, i, fmask] = np.vstack(pv_mat)

    return out_st, np.log10(np.clip(out_pv, 1e-300, 1))


def aug(in_):
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


class Ens(torch.nn.Module):
    def __init__(self, subnet, in_dim, inaux_dim, out_dim):
        super().__init__()
        self.nb_feat = out_dim
        self.main = subnet(in_dim, inaux_dim, out_dim)

    def forward(self, in_, inaux, out=None):
        pred = self.main(in_, inaux)
        if out is None:
            return pred

        mask = ~torch.isnan(out)
        loss = F.mse_loss(pred[mask], out[mask])
        return pred, loss

    def fit(self, data, val_data, opt, steps):
        all_in, all_out = aug(data['in'])
        val_in, val_out = val_data['in'], val_data['in'][1:]
        losses, val_losses = [], []
        for it in range(1, 1+steps):
            opt.zero_grad()
            _, loss = self(all_in, data['inaux'], all_out)
            if it % 50 == 0:
                self.eval()
                val_losses.append(self(val_in, val_data['inaux'], val_out)[1].item())
                self.train()
                losses.append(loss.item())
                if len(val_losses) > 3 and np.gradient(val_losses)[-3:].mean() > 0:
                    print(f'early stopping at {it} iteration')
                    break
            loss.backward()
            opt.step()
        return np.asarray(losses).reshape(1, -1), np.asarray(val_losses).reshape(1, -1)


    def get_mse(self, data):
        def _aws(pred, true):
            return F.mse_loss(pred, true, reduction='none').nanmean(dim=0).cpu().numpy()

        with torch.no_grad():
            all_in, all_out = aug(data['in'])
            mse = np.array_split(_aws(self.main(all_in, data['inaux']), all_out), self.nb_feat+1)
            main_mse = mse[0]
            masks = [mask_1hot(self.nb_feat, i) for i in range(self.nb_feat)]
            aux_mse = [m[:, t] for m, t in zip(mse[1:], masks)]

        return main_mse, aux_mse

    def test_mse(self, main_mse, stack):
        return _test_mse_one2one(self.nb_feat, main_mse, stack)


def get_model(in_dim, inaux_dim, out_dim):
    subnet = lambda i, a, o: model.MinimalRNN(nb_measures=i, h_size=256, nb_layers=1)

    return Ens(subnet, in_dim, inaux_dim, out_dim)
