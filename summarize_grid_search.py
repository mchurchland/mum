#!/usr/bin/env python3
"""Print the best grid-search results from grid_search_ddm.py outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", default="grid_results")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--sort-by",
        default="combined_nll_per_trial",
        choices=["combined_nll_per_trial", "combined_curve_score"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(Path(args.results_dir).glob("grid_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            result = json.loads(line)
            row = {
                "grid_index": result["grid_index"],
                "combined_nll_per_trial": result["combined_nll_per_trial"],
                "combined_curve_score": result["combined_curve_score"],
                **result["params"],
            }
            for label, fit in result["fits"].items():
                row[f"{label}_nll_per_trial"] = fit["nll_per_trial"]
                row[f"{label}_choice_rmse"] = fit["choice_rmse"]
                row[f"{label}_rt_rmse"] = fit["rt_rmse"]
                row[f"{label}_rt_rmse_negative"] = fit["rt_rmse_negative"]
                row[f"{label}_rt_rmse_positive"] = fit["rt_rmse_positive"]
            rows.append(row)

    if not rows:
        raise SystemExit(f"No grid_*.jsonl files found in {args.results_dir}")

    df = pd.DataFrame(rows).sort_values(args.sort_by).head(args.top)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
