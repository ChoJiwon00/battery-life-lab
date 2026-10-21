"""수명 예측 모델 두 가지. 둘 다 log10(수명)을 예측한다.

- 단순 모델: log10 Var(ΔQ₁₀₀₋₁₀) 하나로 직선을 맞춘다 (논문의 variance 모델과 같은 형태)
- 6개 특징 모델: 특징 6개를 표준화한 뒤 Elastic Net (논문의 discharge 모델을 참고)
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, SIMPLE

MIN_TRAIN = {"simple": 3, "full": 6}  # 이보다 적으면 만들지 않는다 (6개 특징을 3~5개 셀로 맞추는 건 의미가 없다)
NAMES = {"simple": "단순 모델 (ΔQ 분산 1개)", "full": "6개 특징 모델 (Elastic Net)"}


class SimpleModel:
    def fit(self, X: pd.DataFrame, life: np.ndarray) -> "SimpleModel":
        self.slope, self.intercept = np.polyfit(X[SIMPLE[0]].to_numpy(), np.log10(life), 1)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return 10 ** (self.slope * X[SIMPLE[0]].to_numpy() + self.intercept)


class FullModel:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X: pd.DataFrame, life: np.ndarray) -> "FullModel":
        folds = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=self.seed)
        self.model = make_pipeline(
            StandardScaler(),
            ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0], cv=folds, max_iter=50_000),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(X[list(FEATURES)].to_numpy(), np.log10(life))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return 10 ** self.model.predict(X[list(FEATURES)].to_numpy())


def make(kind: str, seed: int = 0):
    return SimpleModel() if kind == "simple" else FullModel(seed)


def errors(life: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """평균 % 오차(MAPE)와 RMSE(사이클)."""
    life, pred = np.asarray(life, float), np.asarray(pred, float)
    return {"mape": float(np.mean(np.abs(pred - life) / life) * 100),
            "rmse": float(np.sqrt(np.mean((pred - life) ** 2)))}


def baseline(features: pd.DataFrame) -> pd.DataFrame:
    """논문과 같은 나누기(학습 41 · 테스트 43 · 2차 테스트 40)로 두 모델을 만들고 평가한다."""
    train = features[features["split"] == "train"]
    rows = []
    for kind in ("simple", "full"):
        model = make(kind).fit(train, train["cycle_life"].to_numpy())
        for split in ("train", "test", "secondary_test"):
            part = features[features["split"] == split]
            rows.append({"model": kind, "split": split, "cells": len(part),
                         **errors(part["cycle_life"], model.predict(part))})
    return pd.DataFrame(rows)
