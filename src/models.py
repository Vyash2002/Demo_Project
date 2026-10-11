"""
models.py - steps 5-7: the ML models, the calibrated wrapper that is saved to disk, and SHAP.

Classification ("will it fail in 7 days?"):  logistic regression, random forest, XGBoost.
Regression ("how many days are left?"):      gradient boosting with quantile loss (10/50/90%).
Calibration:                                 Platt scaling (a 1-feature logistic regression).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import SEED
from .features import FEATURES

CLASSIFIERS = ["logistic_regression", "random_forest", "xgboost"]
TTF_QUANTILES = (0.1, 0.5, 0.9)      # low end of range, estimate, high end of range


def make_classifier(name: str, train: pd.DataFrame):
    """Return an untrained failure classifier.

    Class imbalance (~5% failures) is handled with class weights. For XGBoost the weight is
    computed from `train`, i.e. inside the current training fold only.
    Imputation and scaling are part of the sklearn pipeline, so they are also fitted per fold.
    """
    pos_weight = (train.label == 0).sum() / max(train.label.sum(), 1)   # negatives per positive
    return {
        "logistic_regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                                              class_weight="balanced_subsample", random_state=SEED)),
        # XGBoost handles missing values itself, so it needs no imputer.
        "xgboost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=pos_weight, random_state=SEED, n_jobs=-1),
    }[name]


def make_ttf_regressor(q: float):
    """Days-to-failure model for quantile q (0.1 / 0.5 / 0.9 -> range low / estimate / range high)."""
    return make_pipeline(SimpleImputer(strategy="median"),
                         GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=150, max_depth=2,
                                                   random_state=SEED))


def logit(p):
    """Probability -> log-odds, as a column vector (the input of the Platt calibrator)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)            # avoid log(0)
    return np.log(p / (1 - p)).reshape(-1, 1)


class CalibratedModel:
    """Trained classifier + Platt calibrator. This object is what models/model.joblib holds.

    background = a small sample of training rows, needed by SHAP for linear models.
    """

    def __init__(self, model, calibrator, background: pd.DataFrame):
        self.model, self.calibrator, self.background = model, calibrator, background

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated score of the classifier (class weights make it too high on its own)."""
        return self.model.predict_proba(X[FEATURES])[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated failure probability: a predicted 0.8 should mean ~80% of such cases fail."""
        return self.calibrator.predict_proba(logit(self.raw_score(X)))[:, 1]


def shap_values(cm: CalibratedModel, X: pd.DataFrame) -> np.ndarray:
    """SHAP value per row and feature = how much each feature pushed that row's score up or down.

    Values are in the classifier's raw output space (probability for random forest,
    log-odds for logistic regression / XGBoost), before calibration.
    """
    import shap                                   # imported here: slow to import, only needed for explanations
    m = cm.model
    if hasattr(m, "steps"):                      # sklearn pipeline: run imputer (+ scaler), explain the last step
        Xt, est, bg = m[:-1].transform(X[FEATURES]), m[-1], m[:-1].transform(cm.background)
    else:
        Xt, est, bg = X[FEATURES], m, cm.background
    if isinstance(est, LogisticRegression):
        v = shap.LinearExplainer(est, bg).shap_values(Xt)
    else:
        v = shap.TreeExplainer(est).shap_values(Xt)
    # Different SHAP/model versions return one array per class; keep the "failure" class.
    v = np.asarray(v[1] if isinstance(v, list) else v)
    return v[:, :, 1] if v.ndim == 3 else v
