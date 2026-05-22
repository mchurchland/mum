#!/usr/bin/env python3
"""Fit the original PyDDM model with separate left/right drift gains."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyddm import Drift, Fittable, Model, Sample
from pyddm.functions import fit_adjust_model
from pyddm.models import (
    BoundCollapsingExponential,
    InitialCondition,
    NoiseConstant,
    OverlayChain,
    OverlayNonDecisionUniform,
    OverlayPoissonMixture,
)
from pyddm.models.loss import LossRobustLikelihood

warnings.filterwarnings("ignore")
logging.getLogger("pyddm").setLevel(logging.ERROR)


class DriftLeftRightScaled(Drift):
    name = "Left/right scaled drift"
    required_parameters = ["k_left", "k_right", "alpha"]
    required_conditions = ["stim"]

    def get_drift(self, conditions, **kwargs):
        stim = conditions["stim"]
        magnitude = abs(stim) ** self.alpha
        if stim < 0:
            return -self.k_left * magnitude
        return self.k_right * magnitude


class ICPointFrac(InitialCondition):
    name = "Fractional bias"
    required_parameters = ["x0_frac"]

    def get_IC(self, x, dx, conditions):
        bound = max(abs(x))
        if bound == 0:
            bound = dx
        x0 = self.x0_frac * bound
        shift_i = int(round(x0 / dx)) + (len(x) - 1) // 2
        shift_i = int(np.clip(shift_i, 0, len(x) - 1))
        pdf = np.zeros(len(x))
        pdf[shift_i] = 1.0 / dx
        return pdf


def signed_contrast(trials: Any) -> np.ndarray:
    contrast = np.nan_to_num(np.c_[trials["contrastLeft"], trials["contrastRight"]])
    return np.diff(contrast).flatten()


def load_condition(path: Path, block: str | float, sample_size: int | None, seed: int) -> pd.DataFrame:
    with path.open("rb") as f:
        trials = pickle.load(f)

    if block != "all":
        mask = trials["probabilityLeft"] == float(block)
    else:
        mask = np.ones(len(trials["choice"]), dtype=bool)

    df = pd.DataFrame(
        {
            "rt": np.asarray(trials["response_times"] - trials["stimOn_times"])[mask],
            "stim": signed_contrast(trials)[mask],
            "choice": (np.asarray(trials["choice"])[mask] == -1).astype(int),
        }
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df["rt"] > 0) & (df["rt"] <= 1.5)]

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)

    return df.reset_index(drop=True)


def make_model(dx: float, dt: float, t_dur: float) -> Model:
    return Model(
        drift=DriftLeftRightScaled(
            k_left=Fittable(minval=0, maxval=20),
            k_right=Fittable(minval=0, maxval=20),
            alpha=Fittable(minval=0.1, maxval=1.0),
        ),
        noise=NoiseConstant(noise=1),
        bound=BoundCollapsingExponential(
            B=Fittable(minval=0.1, maxval=0.8),
            tau=Fittable(minval=0.05, maxval=2.0),
        ),
        overlay=OverlayChain(
            overlays=[
                OverlayNonDecisionUniform(
                    nondectime=Fittable(minval=0.05, maxval=0.4),
                    halfwidth=Fittable(minval=0.0, maxval=0.2),
                ),
                OverlayPoissonMixture(pmixturecoef=0.05, rate=1),
            ]
        ),
        IC=ICPointFrac(x0_frac=Fittable(minval=-0.99, maxval=0.99)),
        dx=dx,
        dt=dt,
        T_dur=t_dur,
    )


def fitted_params(model: Model) -> dict[str, float]:
    params = {}
    for component, component_params in model.parameters().items():
        for name, value in component_params.items():
            params[f"{component}.{name}"] = float(value.value if hasattr(value, "value") else value)
    return params


def curve_metrics(df: pd.DataFrame, model: Model) -> tuple[list[dict[str, float]], dict[str, float]]:
    rows = []
    for stim in sorted(df["stim"].unique()):
        subset = df[df["stim"] == stim]
        sol = model.solve(conditions={"stim": float(stim)})
        rows.append(
            {
                "stim": float(stim),
                "n": int(len(subset)),
                "p_right_data": float(subset["choice"].mean()),
                "p_right_model": float(sol.prob("upper_bound")),
                "rt_data": float(subset["rt"].mean()),
                "rt_model": float(sol.mean_rt()),
            }
        )

    curve = pd.DataFrame(rows)
    weights = curve["n"] / curve["n"].sum()
    metrics = {
        "choice_rmse": float(np.sqrt(np.sum(weights * (curve["p_right_model"] - curve["p_right_data"]) ** 2))),
        "rt_rmse": float(np.sqrt(np.sum(weights * (curve["rt_model"] - curve["rt_data"]) ** 2))),
    }
    neg = curve["stim"] < 0
    pos = curve["stim"] > 0
    metrics["rt_rmse_negative"] = float(np.sqrt(np.mean((curve.loc[neg, "rt_model"] - curve.loc[neg, "rt_data"]) ** 2))) if neg.any() else None
    metrics["rt_rmse_positive"] = float(np.sqrt(np.mean((curve.loc[pos, "rt_model"] - curve.loc[pos, "rt_data"]) ** 2))) if pos.any() else None
    return rows, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["no_wiggle", "wiggle"], default="no_wiggle")
    parser.add_argument("--block", default="0.2", help="Probability-left block: 0.2, 0.8, 0.5, or all.")
    parser.add_argument("--wiggle-pkl", default="superMouse_WigglesConc.pkl")
    parser.add_argument("--no-wiggle-pkl", default="superMouse_NoWiggConc.pkl")
    parser.add_argument("--sample-size", type=int, default=5000, help="Use 0 for all trials.")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--dx", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--t-dur", type=float, default=1.5)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.no_wiggle_pkl if args.condition == "no_wiggle" else args.wiggle_pkl)
    block = "all" if args.block == "all" else float(args.block)
    sample_size = None if args.sample_size == 0 else args.sample_size

    df = load_condition(path, block=block, sample_size=sample_size, seed=args.seed)
    sample = Sample.from_pandas_dataframe(df, rt_column_name="rt", choice_column_name="choice")
    model = make_model(dx=args.dx, dt=args.dt, t_dur=args.t_dur)

    fit_model = fit_adjust_model(
        sample=sample,
        model=model,
        lossfunction=LossRobustLikelihood,
        verbose=False,
    )

    curve, metrics = curve_metrics(df, fit_model)
    result = {
        "condition": args.condition,
        "block": args.block,
        "sample_size": sample_size,
        "n_trials": int(len(df)),
        "params": fitted_params(fit_model),
        "metrics": metrics,
        "curve": curve,
    }

    print("\n=== Fitted Parameters ===")
    for key, value in result["params"].items():
        print(f"{key}: {value:.6g}")

    print("\n=== Fit Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.6g}")

    print("\n=== Data vs Model ===")
    print(f"{'stim':>8} {'n':>6} {'p_data':>10} {'p_model':>10} {'rt_data':>10} {'rt_model':>10}")
    for row in curve:
        print(
            f"{row['stim']:8.4f} {row['n']:6d} {row['p_right_data']:10.4f} "
            f"{row['p_right_model']:10.4f} {row['rt_data']:10.4f} {row['rt_model']:10.4f}"
        )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
