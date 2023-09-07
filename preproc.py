import numpy as np
import torch


def ast(mat):
    return torch.as_tensor(mat, dtype=torch.float)


def create_rnn(dataframe, subjects, features, exo_features=None):
    data = []
    D = len(features)
    for rid, sf in dataframe.groupby('RID'):
        if rid not in subjects:
            continue
        sf = sf.sort_values('Year')
        indices = sf['Year'].to_numpy()

        assert sf['Year'].min() == 0
        mat = np.full((sf['Year'].max()+1, D), np.nan)
        mat[indices] = sf[features].to_numpy()
        exo = ast(sf[exo_features].iloc[0].to_numpy()).unsqueeze(dim=0) if exo_features else None

        data.append((ast(mat), exo))

    XX, EXX = list(zip(*data))
    XX = torch.nn.utils.rnn.pad_sequence(XX, padding_value=np.nan)
    EXX = torch.cat(EXX, dim=0).cuda() if exo_features else None

    return {'in': XX.cuda(), 'inaux': EXX}
