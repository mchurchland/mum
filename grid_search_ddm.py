#!/usr/bin/env python3
"""Grid search fixed PyDDM parameter sets for wiggle/no-wiggle data.

Each invocation evaluates one parameter combination. This makes it suitable for
SLURM array jobs:

    python grid_search_ddm.py --grid-index "$SLURM_ARRAY_TASK_ID"

The script prints one JSON object containing the parameters and fit metrics for
each requested condition.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyddm import Drift, Model, Sample
from pyddm.models import (
    BoundCollapsingExponential,
    InitialCondition,
    NoiseConstant,
    OverlayChain,
    OverlayNonDecisionUniform,
    OverlayPoissonMixture,
)
from pyddm.models.loss import LossRobustLikelihood


DEFAULT_GRID = {
    # k_left controls negative signed contrasts; k_right controls positive.
    "k_left": [6.0, 10.0, 14.0],
    "k_right": [6.0, 10.0, 14.0],
    "alpha": [0.4, 0.7, 1.0],
    "B": [0.20, 0.35, 0.50],
    "tau": [0.20, 0.70, 1.50],
    "nondectime": [0.08, 0.16, 0.24],
    "x0_frac": [-0.25, 0.0, 0.25],
}


class DriftLeftRightScaled(Drift):
    """Contrast-scaled drift with separate gains for left and right evidence."""

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
    """Starting-point bias as a fraction of the current bound height."""

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
    """Signed contrast, where negative values are left contrasts."""

    contrast = np.nan_to_num(np.c_[trials["contrastLeft"], trials["contrastRight"]])
    return np.diff(contrast).flatten()


def load_condition(path: Path, block: str | float, sample_size: int | None, seed: int) -> pd.DataFrame:
    with path.open("rb") as f:
        trials = pickle.load(f)

    if block != "all":
        mask = trials["probabilityLeft"] == float(block)
    else:
        mask = np.ones(len(trials["choice"]), dtype=bool)

    rt = np.asarray(trials["response_times"] - trials["stimOn_times"])[mask]
    stim = signed_contrast(trials)[mask]
    choice = (np.asarray(trials["choice"])[mask] == -1).astype(int)

    df = pd.DataFrame({"rt": rt, "stim": stim, "choice": choice})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df["rt"] > 0) & (df["rt"] <= 1.5)]

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)

    return df.reset_index(drop=True)


def build_model(params: dict[str, float], dx: float, dt: float, t_dur: float) -> Model:
    return Model(
        drift=DriftLeftRightScaled(
            k_left=params["k_left"],
            k_right=params["k_right"],
            alpha=params["alpha"],
        ),
        noise=NoiseConstant(noise=1),
        bound=BoundCollapsingExponential(B=params["B"], tau=params["tau"]),
        overlay=OverlayChain(
            overlays=[
                OverlayNonDecisionUniform(
                    nondectime=params["nondectime"],
                    halfwidth=params.get("halfwidth", 0.02),
                ),
                OverlayPoissonMixture(pmixturecoef=0.05, rate=1),
            ]
        ),
        IC=ICPointFrac(x0_frac=params["x0_frac"]),
        dx=dx,
        dt=dt,
        T_dur=t_dur,
    )


def condition_metrics(
    label: str,
    df: pd.DataFrame,
    model: Model,
    compute_likelihood: bool,
) -> dict[str, Any]:
    sample = Sample.from_pandas_dataframe(
        df,
        rt_column_name="rt",
        choice_column_name="choice",
    )

    nll = None
    nll_per_trial = None
    if compute_likelihood:
        loss = LossRobustLikelihood(
            sample=sample,
            required_conditions=model.required_conditions,
            dt=model.dt,
            T_dur=model.T_dur,
        )
        nll = float(loss.loss(model))
        nll_per_trial = nll / len(df)

    rows = []
    for stim in sorted(df["stim"].unique()):
        subset = df[df["stim"] == stim]
        sol = model.solve(conditions={"stim": float(stim)})
        p_model = float(sol.prob("upper_bound"))
        rt_model = float(sol.mean_rt())
        rows.append(
            {
                "stim": float(stim),
                "n": int(len(subset)),
                "p_right_data": float(subset["choice"].mean()),
                "p_right_model": p_model,
                "rt_data": float(subset["rt"].mean()),
                "rt_model": rt_model,
            }
        )

    curve = pd.DataFrame(rows)
    weights = curve["n"] / curve["n"].sum()
    choice_rmse = float(np.sqrt(np.sum(weights * (curve["p_right_model"] - curve["p_right_data"]) ** 2)))
    rt_rmse = float(np.sqrt(np.sum(weights * (curve["rt_model"] - curve["rt_data"]) ** 2)))

    neg = curve["stim"] < 0
    pos = curve["stim"] > 0
    rt_rmse_negative = float(np.sqrt(np.mean((curve.loc[neg, "rt_model"] - curve.loc[neg, "rt_data"]) ** 2))) if neg.any() else None
    rt_rmse_positive = float(np.sqrt(np.mean((curve.loc[pos, "rt_model"] - curve.loc[pos, "rt_data"]) ** 2))) if pos.any() else None

    return {
        "label": label,
        "n_trials": int(len(df)),
        "nll": nll,
        "nll_per_trial": nll_per_trial,
        "choice_rmse": choice_rmse,
        "rt_rmse": rt_rmse,
        "rt_rmse_negative": rt_rmse_negative,
        "rt_rmse_positive": rt_rmse_positive,
        "curve": rows,
    }


def load_grid(grid_json: str | None) -> list[dict[str, float]]:
    grid_spec = DEFAULT_GRID if grid_json is None else json.loads(Path(grid_json).read_text())
    keys = list(grid_spec.keys())
    values = [grid_spec[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-index", type=int, default=None, help="Grid point to evaluate. Defaults to SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--print-grid-size", action="store_true", help="Print number of grid points and exit.")
    parser.add_argument("--grid-json", default=None, help="Path to a JSON grid spec. Keys are parameter names; values are arrays.")
    parser.add_argument("--block", default="0.2", help="Probability-left block: 0.2, 0.8, 0.5, or all.")
    parser.add_argument("--conditions", choices=["both", "wiggle", "no_wiggle"], default="both")
    parser.add_argument("--wiggle-pkl", default="superMouse_WigglesConc.pkl")
    parser.add_argument("--no-wiggle-pkl", default="superMouse_NoWiggConc.pkl")
    parser.add_argument("--sample-size", type=int, default=5000, help="Trials per condition. Use 0 for all trials.")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--dx", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--t-dur", type=float, default=1.5)
    parser.add_argument("--skip-likelihood", action="store_true", help="Only compute curve RMSEs.")
    parser.add_argument("--output-dir", default=None, help="Optional directory for one JSON result per grid index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = load_grid(args.grid_json)

    if args.print_grid_size:
        print(len(grid))
        return

    grid_index = args.grid_index
    if grid_index is None:
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id is None:
            raise SystemExit("Pass --grid-index or run under a SLURM array with SLURM_ARRAY_TASK_ID.")
        grid_index = int(task_id)

    if grid_index < 0 or grid_index >= len(grid):
        raise SystemExit(f"grid-index {grid_index} is outside 0..{len(grid) - 1}")

    params = {k: float(v) for k, v in grid[grid_index].items()}
    model = build_model(params, dx=args.dx, dt=args.dt, t_dur=args.t_dur)
    block = "all" if args.block == "all" else float(args.block)
    sample_size = None if args.sample_size == 0 else args.sample_size

    requested = []
    if args.conditions in {"both", "no_wiggle"}:
        requested.append(("no_wiggle", Path(args.no_wiggle_pkl)))
    if args.conditions in {"both", "wiggle"}:
        requested.append(("wiggle", Path(args.wiggle_pkl)))

    metrics = {}
    for offset, (label, path) in enumerate(requested):
        df = load_condition(path, block=block, sample_size=sample_size, seed=args.seed + offset)
        metrics[label] = condition_metrics(
            label=label,
            df=df,
            model=model,
            compute_likelihood=not args.skip_likelihood,
        )

    nll_parts = [m["nll_per_trial"] for m in metrics.values() if m["nll_per_trial"] is not None]
    combined_nll_per_trial = float(np.mean(nll_parts)) if nll_parts else None
    combined_curve_score = float(np.mean([m["choice_rmse"] + m["rt_rmse"] for m in metrics.values()]))

    result = {
        "grid_index": grid_index,
        "grid_size": len(grid),
        "block": args.block,
        "sample_size": sample_size,
        "params": params,
        "combined_nll_per_trial": combined_nll_per_trial,
        "combined_curve_score": combined_curve_score,
        "fits": metrics,
    }

    line = json.dumps(result, sort_keys=True)
    print(line, flush=True)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"grid_{grid_index:06d}.jsonl"
        output_path.write_text(line + "\n")


if __name__ == "__main__":
    main()
