import os
import time
from typing import Optional, Dict, Any, List, Type

from pyscipopt import Model, Branchrule


def solve_mps(
    mps_path: str,
    branchrule_cls: Optional[Type[Branchrule]] = None,
    time_limit: Optional[float] = None,
    use_experimental_setup: bool = True,
    display_verblevel: int = 0,
) -> Dict[str, Any]:
    """
    Read one MPS instance, solve it with optional custom branchrule,
    and return core metrics.
    """
    if not os.path.exists(mps_path):
        raise FileNotFoundError(f"MPS file not found: {mps_path}")

    model = Model()
    model.readProblem(mps_path)

    model.setIntParam("display/verblevel", display_verblevel)

    if time_limit is not None:
        model.setRealParam("limits/time", float(time_limit))

    if use_experimental_setup:
        try:
            model.setIntParam("presolving/maxrounds", 0)
        except Exception:
            pass

        try:
            model.setIntParam("separating/maxrounds", 0)
        except Exception:
            pass

        try:
            model.setIntParam("separating/maxroundsroot", 0)
        except Exception:
            pass

        # try to reduce interference from SCIP default branching rules
        try:
            model.setIntParam("branching/relpscost/priority", -1)
        except Exception:
            pass

        try:
            model.setIntParam("branching/pscost/priority", -1)
        except Exception:
            pass

    branchrule_name = "SCIPDefault"
    if branchrule_cls is not None:
        br = branchrule_cls(model)
        branchrule_name = branchrule_cls.__name__

        priority = getattr(br, "priority", 1000000)
        maxdepth = getattr(br, "maxdepth", -1)
        maxbounddist = getattr(br, "maxbounddist", 1.0)
        name = getattr(br, "name", branchrule_name)
        desc = getattr(br, "desc", f"Custom branchrule {branchrule_name}")

        model.includeBranchrule(
            br,
            name,
            desc,
            priority,
            maxdepth,
            maxbounddist,
        )

    start_cpu = time.process_time()
    start_wall = time.time()

    model.optimize()

    cpu_time = time.process_time() - start_cpu
    wall_time = time.time() - start_wall

    result = {
        "file": os.path.basename(mps_path),
        "path": mps_path,
        "branchrule": branchrule_name,
        "status": str(model.getStatus()),
        "obj": None,
        "dualbound": None,
        "gap": None,
        "nnodes": None,
        "nlps": None,
        "solving_time": None,
        "cpu_process_time": cpu_time,
        "wall_time": wall_time,
        "nsols": None,
    }

    try:
        result["obj"] = model.getObjVal()
    except Exception:
        pass

    try:
        result["dualbound"] = model.getDualbound()
    except Exception:
        pass

    try:
        result["gap"] = model.getGap()
    except Exception:
        pass

    try:
        result["nnodes"] = model.getNNodes()
    except Exception:
        pass

    try:
        result["nlps"] = model.getNLPs()
    except Exception:
        pass

    try:
        result["solving_time"] = model.getSolvingTime()
    except Exception:
        pass

    try:
        result["nsols"] = model.getNSols()
    except Exception:
        pass

    return result


def benchmark_branchrule(
    mps_files: List[str],
    branchrule_cls: Type[Branchrule],
    time_limit: Optional[float] = None,
    use_experimental_setup: bool = True,
    display_verblevel: int = 0,
) -> List[Dict[str, Any]]:
    """
    Benchmark one branchrule class on multiple MPS files.
    Returns one row per instance.
    """
    rows = []

    for mps_path in mps_files:
        print(f"\nRunning {branchrule_cls.__name__} on: {mps_path}")
        try:
            row = solve_mps(
                mps_path=mps_path,
                branchrule_cls=branchrule_cls,
                time_limit=time_limit,
                use_experimental_setup=use_experimental_setup,
                display_verblevel=display_verblevel,
            )
        except Exception as e:
            row = {
                "file": os.path.basename(mps_path),
                "path": mps_path,
                "branchrule": branchrule_cls.__name__,
                "status": f"error: {type(e).__name__}",
                "obj": None,
                "dualbound": None,
                "gap": None,
                "nnodes": None,
                "nlps": None,
                "solving_time": None,
                "cpu_process_time": None,
                "wall_time": None,
                "nsols": None,
                "error_message": str(e),
            }
        rows.append(row)

    return rows


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate benchmark rows into summary statistics.
    """
    n = len(rows)
    solved_count = 0
    timeout_count = 0
    error_count = 0

    solving_times = []
    node_counts = []
    gaps = []
    nlps_list = []

    for row in rows:
        status = str(row.get("status", "")).lower()

        if "optimal" in status:
            solved_count += 1
        if "timelimit" in status:
            timeout_count += 1
        if "error" in status:
            error_count += 1

        t = row.get("solving_time")
        if isinstance(t, (int, float)):
            solving_times.append(float(t))

        nnodes = row.get("nnodes")
        if isinstance(nnodes, (int, float)):
            node_counts.append(float(nnodes))

        gap = row.get("gap")
        if isinstance(gap, (int, float)):
            gaps.append(float(gap))

        nlps = row.get("nlps")
        if isinstance(nlps, (int, float)):
            nlps_list.append(float(nlps))

    def safe_mean(vals):
        return sum(vals) / len(vals) if vals else None

    summary = {
        "n_instances": n,
        "branchrule": rows[0].get("branchrule") if rows else None,
        "solved_count": solved_count,
        "timeout_count": timeout_count,
        "error_count": error_count,
        "avg_solving_time": safe_mean(solving_times),
        "avg_nodes": safe_mean(node_counts),
        "avg_gap": safe_mean(gaps),
        "avg_nlps": safe_mean(nlps_list),
        "raw_rows": rows,
    }
    return summary


def benchmark_default(
    mps_files: List[str],
    time_limit: Optional[float] = None,
    use_experimental_setup: bool = True,
    display_verblevel: int = 0,
) -> List[Dict[str, Any]]:
    """
    Benchmark SCIP default branching on multiple MPS files.
    """
    rows = []
    for mps_path in mps_files:
        print(f"\nRunning SCIP default on: {mps_path}")
        try:
            row = solve_mps(
                mps_path=mps_path,
                branchrule_cls=None,
                time_limit=time_limit,
                use_experimental_setup=use_experimental_setup,
                display_verblevel=display_verblevel,
            )
        except Exception as e:
            row = {
                "file": os.path.basename(mps_path),
                "path": mps_path,
                "branchrule": "SCIPDefault",
                "status": f"error: {type(e).__name__}",
                "obj": None,
                "dualbound": None,
                "gap": None,
                "nnodes": None,
                "nlps": None,
                "solving_time": None,
                "cpu_process_time": None,
                "wall_time": None,
                "nsols": None,
                "error_message": str(e),
            }
        rows.append(row)
    return rows