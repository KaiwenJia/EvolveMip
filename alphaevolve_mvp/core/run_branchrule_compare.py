import argparse
import json
from typing import List, Dict, Any, Optional, Type

from pyscipopt import Branchrule

from baseline_branchrule import BaselinePseudoCostBranchrule
from branchrule_eval import benchmark_branchrule, benchmark_default, summarize_results
from reward import compare_results, compute_reward
from manual_branchrules import (
    PseudoCostSumBranchrule,
    PseudoCostProductBranchrule,
    FractionalityWeightedBranchrule,
)


CANDIDATE_REGISTRY = {
    "sum": PseudoCostSumBranchrule,
    "product": PseudoCostProductBranchrule,
    "fractionality": FractionalityWeightedBranchrule,
}


def load_pool(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=str, required=True)
    parser.add_argument("--candidate", type=str, choices=sorted(CANDIDATE_REGISTRY.keys()), required=True)
    parser.add_argument("--time_limit", type=float, default=30.0)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    mps_files = load_pool(args.pool)
    candidate_cls = CANDIDATE_REGISTRY[args.candidate]

    default_rows = benchmark_default(
        mps_files,
        time_limit=args.time_limit,
    )
    baseline_rows = benchmark_branchrule(
        mps_files,
        branchrule_cls=BaselinePseudoCostBranchrule,
        time_limit=args.time_limit,
    )
    candidate_rows = benchmark_branchrule(
        mps_files,
        branchrule_cls=candidate_cls,
        time_limit=args.time_limit,
    )

    default_summary = summarize_results(default_rows)
    baseline_summary = summarize_results(baseline_rows)
    candidate_summary = summarize_results(candidate_rows)

    vs_baseline = compare_results(baseline_rows, candidate_rows)
    reward_vs_baseline = compute_reward(baseline_rows, candidate_rows)

    vs_default = compare_results(default_rows, candidate_rows)
    reward_vs_default = compute_reward(default_rows, candidate_rows)

    payload = {
        "pool": args.pool,
        "candidate_name": args.candidate,
        "time_limit": args.time_limit,
        "default_summary": default_summary,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "candidate_vs_baseline": vs_baseline,
        "candidate_vs_baseline_reward": reward_vs_baseline,
        "candidate_vs_default": vs_default,
        "candidate_vs_default_reward": reward_vs_default,
        "default_rows": default_rows,
        "baseline_rows": baseline_rows,
        "candidate_rows": candidate_rows,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "candidate_name": args.candidate,
        "candidate_vs_baseline_reward": reward_vs_baseline,
        "candidate_vs_default_reward": reward_vs_default,
        "candidate_summary": candidate_summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()