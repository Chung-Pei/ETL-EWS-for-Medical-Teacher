# -*- coding: utf-8 -*-
"""Student-cluster bootstrap with the same draw mechanism as ews_lib.Boot (cluster = anon_id, shared draws)."""
import numpy as np

import h_lib as H

B_DEFAULT = 2000
SEED_DEV = 20261006
SEED_HOLDOUT = 20261017


class ClusterBoot:
    def __init__(self, anon_ids, B=B_DEFAULT, seed=SEED_DEV):
        uniq, inv = np.unique(np.asarray(anon_ids), return_inverse=True)
        self.rows = [np.where(inv == i)[0] for i in range(len(uniq))]
        draws = np.random.default_rng(seed).integers(0, len(uniq), size=(B, len(uniq)))
        self.idx = [np.concatenate([self.rows[c] for c in d]) for d in draws]
        self.B = B

    def apply(self, fn):
        """fn(ix) -> scalar or 1-d array; returns array of shape (B, ...)."""
        return np.array([fn(ix) for ix in self.idx])

    def auc(self, y, score):
        y = np.asarray(y, int)
        s = np.asarray(score, float)
        return self.apply(lambda ix: H.auc_fail(y[ix], s[ix]))

    def corr(self, score, final):
        s = np.asarray(score, float)
        f = np.asarray(final, float)
        return self.apply(lambda ix: np.corrcoef(s[ix], f[ix])[0, 1])
