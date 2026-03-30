from typing import Dict, Any, List


def _index_by_file(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {r["file"]: r for r in rows}


def _safe_num(x):
    return x if isinstance(x, (int, float)) else None


def _get_eval_time(row: Dict[str, Any]):
    t = row.get("cpu_process_time")
    if isinstance(t, (int, float)):
        return float(t)

    t = row.get("solving_time")
    if isinstance(t, (int, float)):
        return float(t)

    return None


def _status_type(status: str) -> str:
    s = str(status).lower()
    if "error" in s:
        return "error"
    if "optimal" in s:
        return "solved"
    if "timelimit" in s:
        return "timeout"
    return "other"


def compare_results(base_rows: List[Dict[str, Any]], cand_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    base_map = _index_by_file(base_rows)
    cand_map = _index_by_file(cand_rows)

    common_files = sorted(set(base_map) & set(cand_map))

    paired = []
    for fn in common_files:
        b = base_map[fn]
        c = cand_map[fn]
        paired.append({
            "file": fn,
            "base_status": b.get("status"),
            "cand_status": c.get("status"),
            "base_time": _get_eval_time(b),
            "cand_time": _get_eval_time(c),
            "base_nodes": b.get("nnodes"),
            "cand_nodes": c.get("nnodes"),
            "base_gap": b.get("gap"),
            "cand_gap": c.get("gap"),
        })

    return {
        "n_common": len(common_files),
        "paired_rows": paired,
    }


def compute_reward(base_rows: List[Dict[str, Any]], cand_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    base_map = _index_by_file(base_rows)
    cand_map = _index_by_file(cand_rows)
    common_files = sorted(set(base_map) & set(cand_map))

    instance_rewards = []
    solved_gain = 0
    timeout_gain = 0
    error_count = 0

    for fn in common_files:
        b = base_map[fn]
        c = cand_map[fn]

        b_type = _status_type(b.get("status", ""))
        c_type = _status_type(c.get("status", ""))

        bt = _get_eval_time(b)
        ct = _get_eval_time(c)
        bn = _safe_num(b.get("nnodes"))
        cn = _safe_num(c.get("nnodes"))
        bg = _safe_num(b.get("gap"))
        cg = _safe_num(c.get("gap"))

        status_term = 0.0
        if c_type == "solved" and b_type == "timeout":
            status_term = 1.0
            solved_gain += 1
            timeout_gain += 1
        elif c_type == "timeout" and b_type == "solved":
            status_term = -1.0
            solved_gain -= 1
            timeout_gain -= 1
        elif c_type == "error":
            status_term = -2.0
            error_count += 1

        time_term = 0.0
        if bt is not None and ct is not None and bt > 1e-9:
            time_term = (bt - ct) / bt

        node_term = 0.0
        if bn is not None and cn is not None and bn > 0:
            node_term = (bn - cn) / bn

        gap_term = 0.0
        if b_type == "timeout" and c_type == "timeout" and bg is not None and cg is not None and bg > 1e-9:
            gap_term = (bg - cg) / bg

        reward = 0.55 * status_term + 0.25 * time_term + 0.15 * node_term + 0.05 * gap_term
        instance_rewards.append({
            "file": fn,
            "reward": reward,
            "status_term": status_term,
            "time_term": time_term,
            "node_term": node_term,
            "gap_term": gap_term,
            "base_eval_time": bt,
            "cand_eval_time": ct,
        })

    mean_reward = sum(r["reward"] for r in instance_rewards) / len(instance_rewards) if instance_rewards else None

    return {
        "n_common": len(common_files),
        "mean_reward": mean_reward,
        "solved_gain": solved_gain,
        "timeout_gain": timeout_gain,
        "error_count": error_count,
        "instance_rewards": instance_rewards,
        "time_metric": "cpu_process_time_fallback_to_solving_time",
    }