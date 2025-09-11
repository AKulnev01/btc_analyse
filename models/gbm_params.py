# models/gbm_params.py

# общая база для регрессии (квантили)
LGB_PARAMS_REG = dict(
    n_estimators=4000,
    learning_rate=0.05,
    num_leaves=255,
    max_depth=-1,
    min_data_in_leaf=50,
    min_sum_hessian_in_leaf=1e-3,
    feature_fraction=0.9,   # используем ТОЛЬКО это (без colsample_bytree)
    subsample=0.9,
    subsample_freq=1,
    reg_lambda=5.0,
    min_split_gain=0.0,     # вместо min_gain_to_split
    max_bin=511,
    force_row_wise=True,    # убирает ворнинг про выбор row/col wise
    verbosity=-1,
)

# и для классификации (бинарные головы)
LGB_PARAMS_CLF = dict(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=-1,
    min_data_in_leaf=50,
    min_sum_hessian_in_leaf=1e-3,
    feature_fraction=0.9,
    subsample=0.9,
    subsample_freq=1,
    reg_lambda=5.0,
    min_split_gain=0.0,
    max_bin=511,
    force_row_wise=True,
    objective='binary',
    n_jobs=-1,
    verbosity=-1,
)