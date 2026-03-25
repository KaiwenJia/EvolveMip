import os
import csv
import json
import math
import random
import argparse
from typing import List, Dict, Any, Optional

from baseline_branchrule import BaselinePseudoCostBranchrule
from branchrule_eval import benchmark_branchrule


def discover_mps_files(root_dir: str, recursive: bool = True) -> List[str]:
    mps_files = []
    if recursive:
        for root, _, files in os.walk(root_dir):
            for fn in files:
                if fn.lower().endswith(".mps"):
                    mps_files.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(root_dir):
            full = os.path.join(root_dir, fn)
            if os.path.isfile(full) and fn.lower().endswith(".mps"):
                mps_files.append(full)
    return sorted(mps_files)


def safe_float(x) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    return None


def classify_instance(row: Dict[str, Any]) -> str:
    status = str(row.get("status", "")).lower()
    t = safe_float(row.get("solving_time"))
    nnodes = safe_float(row.get("nnodes"))
    gap = safe_float(row.get("gap"))

    if "error" in status:
        return "error"

    if nnodes is not None and nnodes <= 1 and t is not None and t < 0.01:
        return "too_easy"

    if "optimal" in status:
        if nnodes is not None and nnodes <= 3 and t is not None and t < 0.05:
            return "easy"
        if nnodes is not None and nnodes <= 100 and t is not None and t < 1.0:
            return "medium"
        return "hard"

    if "timelimit" in status:
        if gap is not None and gap < 0.05:
            return "hard_timeout_small_gap"
        return "hard_timeout"

    return "other"


def difficulty_score(row: Dict[str, Any]) -> float:
    status = str(row.get("status", "")).lower()
    t = safe_float(row.get("solving_time")) or 0.0
    nnodes = safe_float(row.get("nnodes")) or 0.0
    nlps = safe_float(row.get("nlps")) or 0.0
    gap = safe_float(row.get("gap")) or 0.0

    # 用对数缩放，避免极大实例完全统治排序
    score = 0.0
    score += math.log1p(max(t, 0.0)) * 3.0
    score += math.log1p(max(nnodes, 0.0)) * 2.5
    score += math.log1p(max(nlps, 0.0)) * 1.0

    if "timelimit" in status:
        score += 5.0
        score += min(gap, 1.0) * 3.0

    if "error" in status:
        score = -1e9

    return score


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            pass
        return

    keys = set()
    for r in rows:
        keys.update(r.keys())
    fieldnames = sorted(keys)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_list(path: str, items: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(x + "\n")


def chunked(lst: List[str], batch_size: int) -> List[List[str]]:
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


def run_profiling(
    mps_files: List[str],
    time_limit: float,
    batch_size: int = 50,
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []

    batches = chunked(mps_files, batch_size)
    total = len(mps_files)
    done = 0

    for bi, batch in enumerate(batches, start=1):
        print(f"\n=== Profiling batch {bi}/{len(batches)}: {len(batch)} instances ===")
        rows = benchmark_branchrule(
            batch,
            branchrule_cls=BaselinePseudoCostBranchrule,
            time_limit=time_limit,
        )

        for r in rows:
            r["profile_class"] = classify_instance(r)
            r["difficulty_score"] = difficulty_score(r)

        all_rows.extend(rows)
        done += len(batch)
        print(f"Progress: {done}/{total}")

    return all_rows


def summarize_profile(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for r in rows:
        c = r.get("profile_class", "unknown")
        counts[c] = counts.get(c, 0) + 1

    return {
        "n_instances": len(rows),
        "class_counts": counts,
    }


def select_pools(
    rows: List[Dict[str, Any]],
    dev_size: int,
    holdout_size: int,
    seed: int = 42,
) -> Dict[str, List[str]]:
    rng = random.Random(seed)

    valid_rows = [r for r in rows if r.get("profile_class") not in ("error", "too_easy")]
    if not valid_rows:
        return {"dev_pool": [], "holdout_pool": []}

    # 分桶，尽量保证池子里有不同难度实例
    buckets = {
        "easy": [],
        "medium": [],
        "hard": [],
        "hard_timeout_small_gap": [],
        "hard_timeout": [],
        "other": [],
    }

    for r in valid_rows:
        c = r.get("profile_class", "other")
        buckets.setdefault(c, []).append(r)

    for b in buckets.values():
        b.sort(key=lambda x: x.get("difficulty_score", -1e9), reverse=True)

    # 开发池优先：medium + hard + timeout
    dev_candidates = (
        buckets["medium"] +
        buckets["hard"] +
        buckets["hard_timeout_small_gap"] +
        buckets["hard_timeout"] +
        buckets["easy"] +
        buckets["other"]
    )

    # 去重并按难度排序
    seen = set()
    dev_ranked = []
    for r in dev_candidates:
        p = r["path"]
        if p not in seen:
            seen.add(p)
            dev_ranked.append(r)

    dev_pool = [r["path"] for r in dev_ranked[:dev_size]]

    # holdout 从剩余样本里按分桶抽样
    remaining = [r for r in valid_rows if r["path"] not in set(dev_pool)]

    holdout_buckets = {}
    for r in remaining:
        c = r.get("profile_class", "other")
        holdout_buckets.setdefault(c, []).append(r)

    for c in holdout_buckets:
        rng.shuffle(holdout_buckets[c])

    holdout_pool = []
    classes = ["medium", "hard", "hard_timeout_small_gap", "hard_timeout", "easy", "other"]

    while len(holdout_pool) < holdout_size:
        progressed = False
        for c in classes:
            bucket = holdout_buckets.get(c, [])
            while bucket:
                r = bucket.pop()
                p = r["path"]
                if p not in holdout_pool:
                    holdout_pool.append(p)
                    progressed = True
                    break
            if len(holdout_pool) >= holdout_size:
                break
        if not progressed:
            break

    return {
        "dev_pool": dev_pool,
        "holdout_pool": holdout_pool,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile and select set cover MPS instances.")
    parser.add_argument("--mps_dir", type=str, required=True, help="Root directory containing .mps files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--time_limit", type=float, default=5.0, help="Profiling time limit per instance")
    parser.add_argument("--dev_size", type=int, default=50, help="Development pool size")
    parser.add_argument("--holdout_size", type=int, default=150, help="Holdout pool size")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for profiling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--non_recursive", action="store_true", help="Do not recursively scan mps_dir")
    parser.add_argument("--sample_size", type=int, default=200, help="Randomly sample at most this many MPS files before profiling")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    mps_files = discover_mps_files(args.mps_dir, recursive=not args.non_recursive)
    if not mps_files:
        raise ValueError(f"No .mps files found under: {args.mps_dir}")

    print(f"Discovered {len(mps_files)} MPS files")
    rng = random.Random(args.seed)

    if args.sample_size is not None and args.sample_size > 0 and len(mps_files) > args.sample_size:
        mps_files = rng.sample(mps_files, args.sample_size)

    mps_files = sorted(mps_files)
    print(f"Using {len(mps_files)} sampled MPS files")
    
    rows = run_profiling(
        mps_files=mps_files,
        time_limit=args.time_limit,
        batch_size=args.batch_size,
    )

    profile_summary = summarize_profile(rows)
    pools = select_pools(
        rows=rows,
        dev_size=args.dev_size,
        holdout_size=args.holdout_size,
        seed=args.seed,
    )

    json_path = os.path.join(args.out_dir, "instance_profile.json")
    csv_path = os.path.join(args.out_dir, "instance_profile.csv")
    summary_path = os.path.join(args.out_dir, "profile_summary.json")
    dev_path = os.path.join(args.out_dir, "dev_pool.txt")
    holdout_path = os.path.join(args.out_dir, "holdout_pool.txt")

    write_json(json_path, rows)
    write_csv(csv_path, rows)
    write_json(summary_path, profile_summary)
    write_list(dev_path, pools["dev_pool"])
    write_list(holdout_path, pools["holdout_pool"])

    print("\n=== Done ===")
    print(f"Profile JSON:   {json_path}")
    print(f"Profile CSV:    {csv_path}")
    print(f"Summary JSON:   {summary_path}")
    print(f"Dev pool:       {dev_path} ({len(pools['dev_pool'])} instances)")
    print(f"Holdout pool:   {holdout_path} ({len(pools['holdout_pool'])} instances)")
    print(json.dumps(profile_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()