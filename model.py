#!/usr/bin/env python
import torch
import torch.nn as nn

from rnn import MinimalRNNCell


def jozefowicz_init(forget_gate):
    """
    Initialize the forget gaste bias to 1
    Args:
        forget_gate: forget gate bias term
    References: https://arxiv.org/abs/1602.02410
    """
    forget_gate.data.fill_(1)


class RnnModelInterp(nn.Module):
    """
    Recurrent neural network (RNN) base class
    Missing values (i.e. NaN) are filled using model prediction
    """

    def __init__(self, celltype, nb_measures, h_size, h_drop=.0, i_drop=.0, nb_layers=1):
        super().__init__()
        self.h_ratio = 1. - h_drop
        self.i_ratio = 1. - i_drop

        self.cells = nn.ModuleList()
        self.cells.append(celltype(nb_measures, h_size))
        for _ in range(1, nb_layers):
            self.cells.append(celltype(h_size, h_size))

        self.hid2out = nn.Linear(h_size, nb_measures)

    def _init_hidden(self, batch_sz):
        raise NotImplementedError

    def _dropout_mask(self, batch_sz):
        dev = next(self.parameters()).device
        i_mask = torch.ones(batch_sz, self.hid2out.out_features, device=dev)
        r_mask = [torch.ones(batch_sz, c.hidden_size, device=dev) for c in self.cells]

        if self.training:
            i_mask.bernoulli_(self.i_ratio)
            for mask in r_mask:
                mask.bernoulli_(self.h_ratio)

        return i_mask, r_mask

    def step(self, i_val, hid, masks):
        raise NotImplementedError

    def forward(self, _val_seq, inaux):
        out_val_seq = []

        hidden = self._init_hidden(_val_seq.shape[1])
        masks = self._dropout_mask(_val_seq.shape[1])

        val_seq = _val_seq.clone()
        val_seq[0][torch.isnan(val_seq[0])] = 0 # baseline values (= means)
        for i, j in zip(range(len(val_seq)), range(1, len(val_seq))):
            o_val, hidden = self.step(val_seq[i], hidden, masks)

            out_val_seq.append(o_val)

            # fill in the missing features of the next timepoint
            idx = torch.isnan(val_seq[j])
            val_seq[j][idx] = o_val[idx]

        return torch.stack(out_val_seq)


class MinimalRNN(RnnModelInterp):
    """ Minimal RNN """

    def __init__(self, **kwargs):
        super().__init__(MinimalRNNCell, **kwargs)
        for cell in self.cells:
            jozefowicz_init(cell.bias_hh)

    def _init_hidden(self, batch_sz):
        dev = next(self.parameters()).device
        return [torch.zeros(batch_sz, c.hidden_size, device=dev) for c in self.cells]

    def step(self, i_val, hid, masks):
        i_mask, r_mask = masks
        h_t = hid[0].new(i_val) * i_mask

        states = []
        for layer, prev_h, mask in zip(self.cells, hid, r_mask):
            h_t = layer(h_t, prev_h * mask)
            states.append(h_t)

        o_val = self.hid2out(h_t) + h_t.new(i_val)

        return o_val, states
