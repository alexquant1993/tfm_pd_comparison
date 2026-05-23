"""Per-outer-fold nested Optuna tuning for the classical baselines.

Called by the runner ONCE per outer fold (Pass 2). Receives ONLY the outer-
training rows — and crucially, receives them **uncapped**. Caps are recomputed
inside each inner CV split (strict nested preprocessing, spec §4) so that inner-
validation rows never influence the cap thresholds they are later scored against.
"""
from __future__ import annotations
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.preprocessing import fold_safe_iqr_cap

N_TRIALS_DEFAULT = 50
INNER_FOLDS = 5

optuna.logging.set_verbosity(optuna.logging.WARNING)

def _inner_cv_score(
    make_clf_and_fit,
    X_uncapped: pd.DataFrame,
    y: pd.Series,
    types_meta: dict,
    seed: int,
) -> float:
    """5-fold inner CV with caps refit per inner split.

    The cap is computed from inner-train rows only and applied to both inner-
    train and inner-validation. This prevents inner-val from influencing
    its own cap threshold.
    """
    skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X_uncapped, y):
        X_inner_tr_uncapped = X_uncapped.iloc[tr].reset_index(drop=True)
        X_inner_te_uncapped = X_uncapped.iloc[te].reset_index(drop=True)
        # Strict nested cap
        X_inner_tr, X_inner_te, _ = fold_safe_iqr_cap(
            X_inner_tr_uncapped, X_inner_te_uncapped, types_meta,
        )
        y_inner_tr = y.iloc[tr].reset_index(drop=True)
        y_inner_te = y.iloc[te].reset_index(drop=True)
        clf = make_clf_and_fit(X_inner_tr, y_inner_tr)
        p = clf.predict_proba(X_inner_te)[:, 1]
        aucs.append(roc_auc_score(y_inner_te, p))
    return float(np.mean(aucs))

def _cast_nominals_to_category(X: pd.DataFrame, types_meta: dict) -> pd.DataFrame:
    """Cast nominal columns to pandas category dtype. Done inside each inner
    fit so that train/val category levels are consistent within the fit."""
    X = X.copy()
    for c, i in types_meta.items():
        if i["type"] == "nominal" and c in X.columns:
            X[c] = X[c].astype("category")
    return X

def _align_test_categories(X_te: pd.DataFrame, X_tr: pd.DataFrame) -> pd.DataFrame:
    X_te = X_te.copy()
    for c in X_tr.columns:
        if str(X_tr[c].dtype) == "category":
            X_te[c] = X_te[c].astype(pd.CategoricalDtype(categories=X_tr[c].cat.categories))
    return X_te

def _tune_xgb(X_uncapped: pd.DataFrame, y: pd.Series, types_meta: dict,
              seed: int, n_trials: int) -> dict:
    from xgboost import XGBClassifier

    def objective(trial):
        params = dict(
            tree_method="hist", enable_categorical=True, random_state=seed, n_jobs=-1,
            eval_metric="logloss",
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        )
        def make(Xtr_capped, ytr):
            Xtr_cat = _cast_nominals_to_category(Xtr_capped, types_meta)
            clf = XGBClassifier(**params)
            clf.fit(Xtr_cat, ytr)
            clf._train_cats = Xtr_cat
            orig_predict_proba = clf.predict_proba
            def predict_proba(Xte):
                Xte_cat = _cast_nominals_to_category(Xte, types_meta)
                Xte_cat = _align_test_categories(Xte_cat, clf._train_cats)
                return orig_predict_proba(Xte_cat)
            clf.predict_proba = predict_proba
            return clf
        return _inner_cv_score(make, X_uncapped, y, types_meta, seed)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def _tune_lgbm(X_uncapped: pd.DataFrame, y: pd.Series, types_meta: dict,
               seed: int, n_trials: int) -> dict:
    from lightgbm import LGBMClassifier
    nominal_cols = [c for c, i in types_meta.items()
                    if i["type"] == "nominal" and c in X_uncapped.columns]
    cat_idx = [X_uncapped.columns.get_loc(c) for c in nominal_cols]

    def objective(trial):
        params = dict(
            random_state=seed, n_jobs=-1, verbosity=-1,
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", -1, 12),
            num_leaves=trial.suggest_int("num_leaves", 8, 128),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        )
        def make(Xtr_capped, ytr):
            Xtr_cat = _cast_nominals_to_category(Xtr_capped, types_meta)
            clf = LGBMClassifier(**params)
            clf.fit(Xtr_cat, ytr, categorical_feature=cat_idx)
            clf._train_cats = Xtr_cat
            orig_predict_proba = clf.predict_proba
            def predict_proba(Xte):
                Xte_cat = _cast_nominals_to_category(Xte, types_meta)
                Xte_cat = _align_test_categories(Xte_cat, clf._train_cats)
                return orig_predict_proba(Xte_cat)
            clf.predict_proba = predict_proba
            return clf
        return _inner_cv_score(make, X_uncapped, y, types_meta, seed)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def _tune_logit(X_uncapped: pd.DataFrame, y: pd.Series, types_meta: dict,
                seed: int, n_trials: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    nominal = [c for c, i in types_meta.items()
               if i["type"] == "nominal" and c in X_uncapped.columns]
    other = [c for c in X_uncapped.columns if c not in nominal]

    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        def make(Xtr_capped, ytr):
            pre = ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal),
                ("num", StandardScaler(), other),
            ])
            pipe = Pipeline([("pre", pre),
                             ("lr", LogisticRegression(solver="lbfgs", max_iter=2000,
                                                       C=C, random_state=seed))])
            pipe.fit(Xtr_capped, ytr)
            return pipe
        return _inner_cv_score(make, X_uncapped, y, types_meta, seed)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def tune_classicals_for_fold(
    X_train_outer: pd.DataFrame,
    y_train_outer: pd.Series,
    types_meta: dict,
    seed: int = 42,
    n_trials: int = N_TRIALS_DEFAULT,
) -> dict[str, dict]:
    """Run Optuna for each of {xgb, lgbm, logit} on the outer-training fold only.

    ``X_train_outer`` MUST be the **uncapped** outer-training fold. Caps are
    recomputed inside each inner CV split so inner-validation rows never
    influence the caps they are later scored against (spec §4 strict nested
    preprocessing).

    Returns:
        {"xgb": best_params, "lgbm": best_params, "logit": best_params}
    """
    return {
        "xgb":   _tune_xgb(X_train_outer, y_train_outer, types_meta, seed, n_trials),
        "lgbm":  _tune_lgbm(X_train_outer, y_train_outer, types_meta, seed, n_trials),
        "logit": _tune_logit(X_train_outer, y_train_outer, types_meta, seed, n_trials),
    }
