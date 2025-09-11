"""
Единый конфиг для параметров LightGBM (регрессия/классификация).

Цели:
- убрать дублирующиеся/конфликтные параметры (feature_fraction vs colsample_bytree и т.п.);
- ослабить регуляризацию, чтобы не было ранних остановок на 1–2 итерации;
- повысить разрешение по бинам признаков.
"""

# Квантильная регрессия (P10/P50/P90)
LGB_PARAMS_REG = dict(
    n_estimators=4000,
    learning_rate=0.03,
    num_leaves=127,
    max_depth=-1,

    # чтобы модель действительно делала сплиты
    min_data_in_leaf=25,
    min_sum_hessian_in_leaf=1e-2,

    # используем только feature_fraction (colsample_bytree не задаём вовсе)
    feature_fraction=0.85,
    subsample=0.9,
    subsample_freq=1,

    reg_lambda=3.0,
    min_split_gain=0.0,       # не используем min_gain_to_split
    max_bin=1023,

    force_row_wise=True,      # снимет лишний оверхед
    verbosity=-1,
)

# Бинарные головы (P(up), P(big_move))
LGB_PARAMS_CLF = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=127,
    max_depth=-1,

    min_data_in_leaf=25,
    min_sum_hessian_in_leaf=1e-2,

    feature_fraction=0.85,
    subsample=0.9,
    subsample_freq=1,

    reg_lambda=3.0,
    min_split_gain=0.0,
    max_bin=1023,

    force_row_wise=True,
    objective='binary',
    n_jobs=-1,
    verbosity=-1,
)