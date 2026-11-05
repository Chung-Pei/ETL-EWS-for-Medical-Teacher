# -*- coding: utf-8 -*-
"""Retraining pipeline shared by G-b, G-c, R7-1 and R7-2.  Uses the pack's own ews_lib / ews_experiment (unmodified) and
ETL v7.15.34 module 11.  Outcome = semester score (frozen target; keyed file final/y, 560 failing), Ruling R1 value for
WITHHELD/1141 (53.2) comes from the keyed file via apply_semester_outcome."""
import r7_common as C
import numpy as np
import pandas as pd

L = C.L


def setup(weeks=None):
    weeks = weeks or C.W_ALL
    EL, EE = C.import_pipeline()
    root = C.merge_step9()
    etl = EL.Etl(str(C.ETL_DIR))
    EL.init_bas(etl)
    base, scores = C.load_base()
    data = {}
    for W in weeks:
        d = EE.load_all(etl, str(root), [W])
        C.apply_semester_outcome(d, base)
        data.update(d)
    n = EE.check_alignment(data)
    assert n == 2652, n
    return EL, EE, etl, base, scores, data


def fit_pair(EE, etl, data_W, train_sems, test_sem, models=None):
    """One fold: train on `train_sems`, score `test_sem` (retaker rule of run_folds: training records of students who
    also appear in the test semester are removed).  Returns the out-of-fold frame (m1, m2, pfail, y, final, anon_id)
    and the number of removed training records."""
    models = models or {'m1': 1, 'm2': 1}
    df = EE.run_folds(etl, data_W, [(test_sem, list(train_sems))], etl.FS_C, False, models)
    return EE.canon(df), int(df.attrs['removed'][test_sem])
