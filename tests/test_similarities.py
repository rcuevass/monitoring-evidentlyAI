import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def similarity(series_ref: np.array, series_prod: np.array) -> float:
    t, p = ttest_ind(series_ref, series_prod)
    return p


def test_trivial_equal():
    x_ = np.random.random(1000)
    assert similarity(x_, x_) == 1.0


def test_trivial_different():
    x_ = np.random.random(1000)
    y_ = 0*x_
    assert similarity(x_, y_) != 1.0


def test_unforced():
    df_ref = pd.read_csv('../data/input/test_reference.csv')
    df_prod_unforced = pd.read_csv('../data/input/test_prod_unforced.csv')
    age_ref = df_ref['age'].values
    age_prod_unforced = df_prod_unforced['age'].values
    assert similarity(age_ref, age_prod_unforced) > 0.5


def test_forced():
    df_ref = pd.read_csv('../data/input/test_reference.csv')
    df_prod_forced = pd.read_csv('../data/input/test_prod_forced.csv')
    age_ref = df_ref['age'].values
    age_prod_forced = df_prod_forced['age'].values
    assert similarity(age_ref, age_prod_forced) < 0.5

