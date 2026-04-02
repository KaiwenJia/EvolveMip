import inspect
import json
from typing import List

from baseline_branchrule import BaselinePseudoCostBranchrule
from branchrule_eval import benchmark_branchrule, summarize_results
from llm_codegen import LLMBranchruleGenerator


# ============================================================
# LLM 配置区：只需要改这里
# ============================================================

LLM_PROVIDER = "deepseek"
# 可选：
# "deepseek"
# "openai"
# "custom_proxy"

LLM_CONFIGS = {
    "deepseek": {
        "api_key": "",
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat",
    },
    "openai": {
        "api_key": "你的_openai_api_key",
        "base_url": None,   # OpenAI 官方可留空
        "model_name": "gpt-4.1-mini",
    },
    "custom_proxy": {
        "api_key": "你的_proxy_api_key",
        "base_url": "https://your-proxy-base-url/v1",
        "model_name": "gemini-2.5-flash",
    },
}


def get_llm_config(provider: str):
    if provider not in LLM_CONFIGS:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Available providers: {list(LLM_CONFIGS.keys())}"
        )

    cfg = LLM_CONFIGS[provider]

    api_key = cfg.get("api_key")
    if not api_key or api_key.startswith("你的_"):
        raise ValueError(
            f"Provider '{provider}' 的 api_key 还没填写。"
        )

    return cfg


def build_feedback(baseline_summary, candidate_summary) -> str:
    lines = []

    lines.append("Benchmark comparison on set cover MPS instances:")
    lines.append(
        f"Baseline: avg_solving_time={baseline_summary.get('avg_solving_time')}, "
        f"avg_nodes={baseline_summary.get('avg_nodes')}, "
        f"solved={baseline_summary.get('solved_count')}, "
        f"timeout={baseline_summary.get('timeout_count')}, "
        f"errors={baseline_summary.get('error_count')}"
    )
    lines.append(
        f"Candidate: avg_solving_time={candidate_summary.get('avg_solving_time')}, "
        f"avg_nodes={candidate_summary.get('avg_nodes')}, "
        f"solved={candidate_summary.get('solved_count')}, "
        f"timeout={candidate_summary.get('timeout_count')}, "
        f"errors={candidate_summary.get('error_count')}"
    )
    lines.append(
        "Goal: reduce solving time and node count on set cover instances "
        "while keeping the branching rule valid, simple, and computationally cheap."
    )
    lines.append(
        "Prefer modifying only the variable scoring logic inside branchexeclp."
    )

    return "\n".join(lines)


def get_mps_files() -> List[str]:
    """
    Replace this with your real set cover benchmark instances.
    You can either hardcode a list or scan a directory.
    """
    mps_files = [
        # Example:
        # "/path/to/instance1.mps",
        # "/path/to/instance2.mps",
    ]

    if not mps_files:
        raise ValueError(
            "No MPS files provided. Please edit get_mps_files() and add real set cover .mps paths."
        )

    return mps_files


def main():
    mps_files = get_mps_files()

    llm_cfg = get_llm_config(LLM_PROVIDER)

    print("=== Using LLM Config ===")
    print(json.dumps({
        "provider": LLM_PROVIDER,
        "base_url": llm_cfg["base_url"],
        "model_name": llm_cfg["model_name"],
    }, indent=2, ensure_ascii=False))

    baseline_rows = benchmark_branchrule(
        mps_files,
        branchrule_cls=BaselinePseudoCostBranchrule,
        time_limit=300.0,
    )
    baseline_summary = summarize_results(baseline_rows)

    print("=== Baseline Summary ===")
    print(json.dumps(baseline_summary, indent=2, ensure_ascii=False))

    baseline_code = inspect.getsource(BaselinePseudoCostBranchrule)

    generator = LLMBranchruleGenerator(
        api_key=llm_cfg["api_key"],
        base_url=llm_cfg["base_url"],
        model_name=llm_cfg["model_name"],
    )

    initial_feedback = (
        "Current baseline performance on set cover MPS instances:\n"
        + json.dumps(baseline_summary, ensure_ascii=False, indent=2)
        + "\nPlease improve over this baseline. "
          "Keep the branchrule valid and only modify branching candidate scoring logic if possible."
    )

    generated_code = generator.generate_code(
        baseline_code=baseline_code,
        feedback_text=initial_feedback,
    )

    print("\n=== Generated Code ===\n")
    print(generated_code)

    result_payload = {
        "llm_provider": LLM_PROVIDER,
        "llm_base_url": llm_cfg["base_url"],
        "llm_model_name": llm_cfg["model_name"],
        "baseline_rows": baseline_rows,
        "baseline_summary": baseline_summary,
        "generated_code": generated_code,
        "candidate_rows": None,
        "candidate_summary": None,
        "next_feedback": None,
        "candidate_error": None,
    }

    try:
        generator.validate_code(generated_code)
        GeneratedCls = generator.load_branchrule_class(generated_code)

        candidate_rows = benchmark_branchrule(
            mps_files,
            branchrule_cls=GeneratedCls,
            time_limit=300.0,
        )
        candidate_summary = summarize_results(candidate_rows)

        print("\n=== Candidate Summary ===")
        print(json.dumps(candidate_summary, indent=2, ensure_ascii=False))

        feedback = build_feedback(baseline_summary, candidate_summary)
        print("\n=== Feedback for Next Round ===")
        print(feedback)

        result_payload["candidate_rows"] = candidate_rows
        result_payload["candidate_summary"] = candidate_summary
        result_payload["next_feedback"] = feedback

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print("\n=== Candidate Evaluation Failed ===")
        print(error_msg)
        result_payload["candidate_error"] = error_msg

    with open("generated_branchrule.py", "w", encoding="utf-8") as f:
        f.write(generated_code)

    with open("llm_eval_result.json", "w", encoding="utf-8") as f:
        json.dump(
            result_payload,
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
